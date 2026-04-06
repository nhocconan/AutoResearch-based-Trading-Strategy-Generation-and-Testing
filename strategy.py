#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray with 12h regime filter
# Elder Ray: Bull Power = High - EMA(13), Bear Power = EMA(13) - Low
# Bull market regime (12h ADX > 25): go long when Bull Power > 0 and rising
# Bear market regime (12h ADX < 20): go short when Bear Power > 0 and rising
# Range market (20 <= 12h ADX <= 25): fade extremes - long when Bear Power < 0 and rising, short when Bull Power < 0 and rising
# Volume confirmation: require volume > 20-period average
# Target: 50-150 total trades over 4 years with balanced performance in bull/bear/range

name = "6h_elderray_12h_regime_v3"
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
    
    # 12h data for regime filter (ADX)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate ADX components
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros(len(high))
        minus_dm = np.zeros(len(high))
        tr = np.zeros(len(high))
        
        for i in range(1, len(high)):
            high_diff = high[i] - high[i-1]
            low_diff = low[i-1] - low[i]
            
            plus_dm[i] = high_diff if high_diff > low_diff and high_diff > 0 else 0
            minus_dm[i] = low_diff if low_diff > high_diff and low_diff > 0 else 0
            
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Smooth with Wilder's smoothing (alpha = 1/period)
        atr = np.zeros(len(high))
        plus_di = np.zeros(len(high))
        minus_di = np.zeros(len(high))
        
        atr[period-1] = np.mean(tr[:period])
        plus_dm_sum = np.sum(plus_dm[:period])
        minus_dm_sum = np.sum(minus_dm[:period])
        
        for i in range(period, len(high)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
            plus_dm_sum = plus_dm_sum - (plus_dm_sum / period) + plus_dm[i]
            minus_dm_sum = minus_dm_sum - (minus_dm_sum / period) + minus_dm[i]
            plus_di[i] = 100 * plus_dm_sum / (atr[i] * period) if atr[i] != 0 else 0
            minus_di[i] = 100 * minus_dm_sum / (atr[i] * period) if atr[i] != 0 else 0
        
        dx = np.zeros(len(high))
        adx = np.zeros(len(high))
        for i in range(period, len(high)):
            if plus_di[i] + minus_di[i] != 0:
                dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
            else:
                dx[i] = 0
        
        adx[2*period-2] = np.mean(dx[period-1:2*period-1])
        for i in range(2*period-1, len(high)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx_12h = calculate_adx(high_12h, low_12h, close_12h, 14)
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # Elder Ray components (13-period EMA)
    ema_13 = pd.Series(close).ewm(span=13, min_periods=13, adjust=False).mean().values
    bull_power = high - ema_13
    bear_power = ema_13 - low
    
    # Volume filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(30, n):
        # Skip if required data not available
        if (np.isnan(adx_12h_aligned[i]) or np.isnan(ema_13[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2 * ATR approximation
            if close[i] < entry_price - 2.0 * (high[i] - low[i]):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit conditions based on regime
            elif adx_12h_aligned[i] > 25:  # Bull trend
                if bull_power[i] <= 0:  # Bull power failed
                    signals[i] = 0.0
                    position = 0
                    entry_price = 0.0
            elif adx_12h_aligned[i] < 20:  # Bear trend
                if bear_power[i] <= 0:  # Bear power failed
                    signals[i] = 0.0
                    position = 0
                    entry_price = 0.0
            else:  # Range
                if bull_power[i] >= 0:  # Bull power recovered
                    signals[i] = 0.0
                    position = 0
                    entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2 * ATR approximation
            if close[i] > entry_price + 2.0 * (high[i] - low[i]):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit conditions based on regime
            elif adx_12h_aligned[i] > 25:  # Bull trend
                if bear_power[i] >= 0:  # Bear power failed
                    signals[i] = 0.0
                    position = 0
                    entry_price = 0.0
            elif adx_12h_aligned[i] < 20:  # Bear trend
                if bull_power[i] >= 0:  # Bull power recovered
                    signals[i] = 0.0
                    position = 0
                    entry_price = 0.0
            else:  # Range
                if bear_power[i] >= 0:  # Bear power recovered
                    signals[i] = 0.0
                    position = 0
                    entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation
            if vol_filter[i]:
                # Regime-based entries
                if adx_12h_aligned[i] > 25:  # Bull trend - buy strength
                    if bull_power[i] > 0 and bull_power[i] > bull_power[i-1]:
                        signals[i] = 0.25
                        position = 1
                        entry_price = close[i]
                elif adx_12h_aligned[i] < 20:  # Bear trend - sell weakness
                    if bear_power[i] > 0 and bear_power[i] > bear_power[i-1]:
                        signals[i] = -0.25
                        position = -1
                        entry_price = close[i]
                else:  # Range - fade extremes
                    if bear_power[i] < 0 and bear_power[i] > bear_power[i-1]:
                        signals[i] = 0.25
                        position = 1
                        entry_price = close[i]
                    elif bull_power[i] < 0 and bull_power[i] > bull_power[i-1]:
                        signals[i] = -0.25
                        position = -1
                        entry_price = close[i]
    
    return signals