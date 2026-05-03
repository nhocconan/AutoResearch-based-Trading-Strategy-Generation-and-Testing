#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter, volume confirmation, and ATR-based stoploss.
# Long: Close > Donchian Upper(20) AND price > 1d EMA50 (uptrend) AND volume > 1.8x 20-period MA
# Short: Close < Donchian Lower(20) AND price < 1d EMA50 (downtrend) AND volume > 1.8x 20-period MA
# Exit: Opposite Donchian breakout OR ATR-based trailing stop (signal→0 when price < highest_high - 2*ATR for longs, or price > lowest_low + 2*ATR for shorts)
# Discrete sizing 0.25. Target: 80-180 total trades over 4 years (20-45/year).
# Donchian channels provide structural breakouts; 1d EMA50 filters higher timeframe trend to avoid counter-trend trades;
# volume confirmation reduces false breakouts; ATR stop manages risk. Works in bull via long breakouts with trend alignment
# and in bear via short breakouts with trend alignment.

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
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Donchian channels (20-period) on 4h data
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate ATR(14) for stoploss
    tr1 = pd.Series(high - low).values
    tr2 = pd.Series(np.abs(high - np.roll(close, 1))).values
    tr3 = pd.Series(np.abs(low - np.roll(close, 1))).values
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume regime: current 4h volume > 1.8x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_high_since_entry = 0.0
    lowest_low_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(atr[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        ema_trend = ema_50_1d_aligned[i]
        upper_channel = highest_high[i]
        lower_channel = lowest_low[i]
        atr_val = atr[i]
        vol_spike = volume_spike[i]
        
        # Determine trend regime
        is_uptrend = close_val > ema_trend
        is_downtrend = close_val < ema_trend
        
        # Update highest/lowest since entry for trailing stop
        if position == 1:
            highest_high_since_entry = max(highest_high_since_entry, high[i])
        elif position == -1:
            lowest_low_since_entry = min(lowest_low_since_entry, low[i])
        
        # Entry logic
        if position == 0:
            # Long: Close > Donchian Upper(20) AND uptrend AND volume spike
            if close_val > upper_channel and is_uptrend and vol_spike:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
                highest_high_since_entry = high[i]
            # Short: Close < Donchian Lower(20) AND downtrend AND volume spike
            elif close_val < lower_channel and is_downtrend and vol_spike:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
                lowest_low_since_entry = low[i]
        elif position == 1:
            # Long exit: Opposite breakout OR ATR trailing stop
            if close_val < lower_channel or close_val < (highest_high_since_entry - 2.0 * atr_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Opposite breakout OR ATR trailing stop
            if close_val > upper_channel or close_val > (lowest_low_since_entry + 2.0 * atr_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals