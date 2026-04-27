#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Choppiness Index regime filter + Donchian(20) breakout + volume confirmation
# In low-chop (trending) markets: follow breakout direction. In high-chop (range) markets: fade extremes.
# Uses 1d ADX for additional trend strength filter. Target: 20-50 trades/year to avoid fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for regime filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX(14) on 1d for trend strength
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            plus_dm[i] = max(high[i] - high[i-1], 0)
            minus_dm[i] = max(low[i-1] - low[i], 0)
            if plus_dm[i] < minus_dm[i]:
                plus_dm[i] = 0
            if minus_dm[i] < plus_dm[i]:
                minus_dm[i] = 0
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Wilder's smoothing
        atr = np.zeros_like(high)
        plus_di = np.zeros_like(high)
        minus_di = np.zeros_like(high)
        dx = np.zeros_like(high)
        
        atr[period] = np.mean(tr[1:period+1])
        plus_dm_sum = np.sum(plus_dm[1:period+1])
        minus_dm_sum = np.sum(minus_dm[1:period+1])
        
        for i in range(period+1, len(high)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
            plus_dm_sum = plus_dm_sum - (plus_dm_sum/period) + plus_dm[i]
            minus_dm_sum = minus_dm_sum - (minus_dm_sum/period) + minus_dm[i]
            plus_di[i] = 100 * plus_dm_sum / atr[i]
            minus_di[i] = 100 * minus_dm_sum / atr[i]
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
        
        adx = np.zeros_like(high)
        adx[2*period] = np.mean(dx[period+1:2*period+1])
        for i in range(2*period+1, len(high)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        return adx
    
    adx_14 = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_14)
    
    # Calculate Choppiness Index(14) on 1d
    def calculate_chop(high, low, close, period=14):
        atr_sum = np.zeros_like(close)
        max_high = np.zeros_like(high)
        min_low = np.zeros_like(low)
        
        for i in range(len(close)):
            if i == 0:
                tr = high[i] - low[i]
            else:
                tr = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
            
            if i < period:
                atr_sum[i] = np.sum([tr for j in range(max(0, i-period+1), i+1)])
                max_high[i] = np.max(high[max(0, i-period+1):i+1])
                min_low[i] = np.min(low[max(0, i-period+1):i+1])
            else:
                atr_sum[i] = atr_sum[i-1] + tr - (high[i-period] - low[i-period] if i-period >= 0 else 0)
                max_high[i] = np.max(high[i-period+1:i+1])
                min_low[i] = np.min(low[i-period+1:i+1])
            
            if max_high[i] - min_low[i] != 0:
                chop = 100 * np.log10(atr_sum[i] / (max_high[i] - min_low[i])) / np.log10(period)
            else:
                chop = 50
            atr_sum[i] = chop
        return atr_sum
    
    chop_14 = calculate_chop(high_1d, low_1d, close_1d, 14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_14)
    
    # Calculate Donchian(20) on 4h
    def calculate_donchian(high, low, period=20):
        upper = np.full_like(high, np.nan)
        lower = np.full_like(high, np.nan)
        for i in range(period-1, len(high)):
            upper[i] = np.max(high[i-period+1:i+1])
            lower[i] = np.min(low[i-period+1:i+1])
        return upper, lower
    
    donch_up, donch_dn = calculate_donchian(high, low, 20)
    
    # Volume filter: volume > 1.5 x 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need 1d data (20 bars for indicators)
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donch_up[i]) or np.isnan(donch_dn[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(chop_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        adx = adx_aligned[i]
        chop = chop_aligned[i]
        
        # Volume filter: significant volume
        vol_filter = vol_now > 1.5 * vol_avg
        
        # Regime filters
        trending = chop < 38.2 and adx > 20  # Strong trend
        ranging = chop > 61.8                # Range/chop
        
        if position == 0:
            if trending and vol_filter:
                # Trend following: break Donchian bands
                if price > donch_up[i]:
                    signals[i] = size
                    position = 1
                elif price < donch_dn[i]:
                    signals[i] = -size
                    position = -1
                else:
                    signals[i] = 0.0
            elif ranging and vol_filter:
                # Mean reversion: fade Donchian extremes
                if price > donch_up[i]:
                    signals[i] = -size  # Short at upper band
                    position = -1
                elif price < donch_dn[i]:
                    signals[i] = size   # Long at lower band
                    position = 1
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: reverse signal or volatility expansion
            if (ranging and price < donch_up[i]) or (trending and price < donch_dn[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: reverse signal or volatility expansion
            if (ranging and price > donch_dn[i]) or (trending and price > donch_up[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Chop_ADX_Donchian_Breakout_MeanRev"
timeframe = "4h"
leverage = 1.0