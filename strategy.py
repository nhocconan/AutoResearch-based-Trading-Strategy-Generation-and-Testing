#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter, volume confirmation, and ATR-based stoploss.
# Long when price breaks above 20-period Donchian high with volume > 1.5x 20-period MA and close > 1d EMA50 (uptrend).
# Short when price breaks below 20-period Donchian low with volume spike and close < 1d EMA50 (downtrend).
# Uses ATR(14) for dynamic stoploss: exit long if price drops 2*ATR from entry, exit short if price rises 2*ATR from entry.
# Discrete sizing 0.25. Target: 75-200 total trades over 4 years (19-50/year).
# Donchian channels provide clear structure; EMA50 filters counter-trend trades in bear markets.
# Volume confirmation reduces false breakouts. Works in bull/bear via trend alignment.

name = "4h_Donchian20_1dEMA50_Volume_ATR"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Donchian(20) channels
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume regime: current 4h volume > 1.5x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    max_high_since_entry = 0.0
    min_low_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(atr[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        ema_trend = ema_50_1d_aligned[i]
        upper = donchian_high[i]
        lower = donchian_low[i]
        atr_val = atr[i]
        vol_spike = volume_spike[i]
        
        # Determine trend regime
        is_uptrend = close_val > ema_trend
        is_downtrend = close_val < ema_trend
        
        # Update tracking variables for stoploss
        if position == 1:
            max_high_since_entry = max(max_high_since_entry, high_val)
        elif position == -1:
            min_low_since_entry = min(min_low_since_entry, low_val)
        
        # Entry logic
        if position == 0:
            # Long: break above Donchian high with volume spike in uptrend
            if close_val > upper and vol_spike and is_uptrend:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
                max_high_since_entry = high_val
            # Short: break below Donchian low with volume spike in downtrend
            elif close_val < lower and vol_spike and is_downtrend:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
                min_low_since_entry = low_val
        elif position == 1:
            # Long exit: price drops 2*ATR from entry high OR trend turns down
            if high_val < (max_high_since_entry - 2.0 * atr_val) or not is_uptrend:
                signals[i] = 0.0
                position = 0
                max_high_since_entry = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price rises 2*ATR from entry low OR trend turns up
            if low_val > (min_low_since_entry + 2.0 * atr_val) or not is_downtrend:
                signals[i] = 0.0
                position = 0
                min_low_since_entry = 0.0
            else:
                signals[i] = -0.25
    
    return signals