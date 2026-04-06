#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Supertrend with 12h/1d ADX filter and volume confirmation
# Supertrend identifies trend direction with ATR-based bands.
# ADX > 25 filters for strong trends, avoiding whipsaws in ranging markets.
# Volume spike confirms institutional participation.
# Designed to work in both bull (trend following) and bear (short trends) markets.
# Target: 80-180 total trades over 4 years with controlled risk/reward.

name = "6h_supertrend_12h_adx_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h data for ADX trend strength filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # ADX(14) calculation on 12h
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros(len(high))
        minus_dm = np.zeros(len(high))
        tr = np.zeros(len(high))
        
        for i in range(1, len(high)):
            plus_dm[i] = max(0, high[i] - high[i-1])
            minus_dm[i] = max(0, low[i-1] - low[i])
            if plus_dm[i] < minus_dm[i]:
                plus_dm[i] = 0
            if minus_dm[i] < plus_dm[i]:
                minus_dm[i] = 0
            
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Wilder's smoothing
        atr = np.zeros(len(high))
        atr[period-1] = np.mean(tr[:period])
        for i in range(period, len(high)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        plus_di = 100 * (np.zeros(len(high)))
        minus_di = 100 * (np.zeros(len(high)))
        
        plus_dm_smooth = np.zeros(len(high))
        minus_dm_smooth = np.zeros(len(high))
        plus_dm_smooth[period-1] = np.sum(plus_dm[:period])
        minus_dm_smooth[period-1] = np.sum(minus_dm[:period])
        
        for i in range(period, len(high)):
            plus_dm_smooth[i] = plus_dm_smooth[i-1] - (plus_dm_smooth[i-1] / period) + plus_dm[i]
            minus_dm_smooth[i] = minus_dm_smooth[i-1] - (minus_dm_smooth[i-1] / period) + minus_dm[i]
        
        plus_di = 100 * plus_dm_smooth / atr
        minus_di = 100 * minus_dm_smooth / atr
        
        dx = np.zeros(len(high))
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        dx[0:period] = np.nan
        
        adx = np.zeros(len(high))
        adx[2*period-1] = np.nanmean(dx[period:2*period])
        for i in range(2*period, len(high)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx_12h = calculate_adx(high_12h, low_12h, close_12h, 14)
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # Supertrend calculation on 6h
    def supertrend(high, low, close, period=10, multiplier=3.0):
        # Calculate ATR
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr2[0] = tr1[0]
        tr3[0] = tr1[0]
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        atr = np.zeros(len(close))
        for i in range(len(close)):
            if i < period:
                atr[i] = np.nan
            elif i == period:
                atr[i] = np.mean(tr[:period+1])
            else:
                atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        # Calculate basic upper and lower bands
        hl2 = (high + low) / 2
        upper_band = hl2 + multiplier * atr
        lower_band = hl2 - multiplier * atr
        
        # Initialize final bands
        final_upper = np.zeros(len(close))
        final_lower = np.zeros(len(close))
        supertrend = np.zeros(len(close))
        direction = np.ones(len(close))  # 1 for uptrend, -1 for downtrend
        
        # Set initial values
        final_upper[0] = upper_band[0]
        final_lower[0] = lower_band[0]
        supertrend[0] = hl2[0]
        direction[0] = 1
        
        for i in range(1, len(close)):
            # Upper band logic
            if upper_band[i] < final_upper[i-1] or close[i-1] > final_upper[i-1]:
                final_upper[i] = upper_band[i]
            else:
                final_upper[i] = final_upper[i-1]
            
            # Lower band logic
            if lower_band[i] > final_lower[i-1] or close[i-1] < final_lower[i-1]:
                final_lower[i] = lower_band[i]
            else:
                final_lower[i] = final_lower[i-1]
            
            # Supertrend and direction
            if i == 1:
                supertrend[i] = final_lower[i]
                direction[i] = 1
            else:
                if supertrend[i-1] == final_upper[i-1]:
                    if close[i] <= final_upper[i]:
                        supertrend[i] = final_upper[i]
                        direction[i] = -1
                    else:
                        supertrend[i] = final_upper[i]
                        direction[i] = 1
                else:
                    if close[i] >= final_lower[i]:
                        supertrend[i] = final_lower[i]
                        direction[i] = 1
                    else:
                        supertrend[i] = final_lower[i]
                        direction[i] = -1
        
        return supertrend, direction, atr
    
    st, direction, atr = supertrend(high, low, close, 10, 3.0)
    
    # Volume average (20-period)
    volume_ma = np.zeros(len(volume))
    for i in range(len(volume)):
        if i < 20:
            volume_ma[i] = np.nan
        elif i == 20:
            volume_ma[i] = np.mean(volume[:21])
        else:
            volume_ma[i] = (volume_ma[i-1] * 19 + volume[i]) / 20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(30, n):  # Start after warmup
        # Skip if required data not available
        if (np.isnan(adx_12h_aligned[i]) or np.isnan(st[i]) or 
            np.isnan(direction[i]) or np.isnan(atr[i]) or np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2 * ATR
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: trend changes or ADX weak
            elif direction[i] == -1 or adx_12h_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2 * ATR
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: trend changes or ADX weak
            elif direction[i] == 1 or adx_12h_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation and strong trend
            # Long: uptrend, ADX strong, volume spike
            if (direction[i] == 1 and 
                adx_12h_aligned[i] > 25 and
                volume[i] > 1.5 * volume_ma[i]):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: downtrend, ADX strong, volume spike
            elif (direction[i] == -1 and 
                  adx_12h_aligned[i] > 25 and
                  volume[i] > 1.5 * volume_ma[i]):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals