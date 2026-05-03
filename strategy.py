#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 12h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation
# Long when price breaks above 12h Donchian upper with 1d EMA50 uptrend and volume > 1.8x 20-bar average
# Short when price breaks below 12h Donchian lower with 1d EMA50 downtrend and volume > 1.8x 20-bar average
# Exit via ATR(14) trailing stop: long exit when price < highest_high_since_entry - 2.0 * ATR
#                      short exit when price > lowest_low_since_entry + 2.0 * ATR
# Uses 12h Donchian channels for structure, 1d EMA50 for trend filter, volume for confirmation
# Discrete sizing 0.25 balances return and fee drag. Target: 75-200 total trades over 4 years = 19-50/year.

name = "4h_Donchian20_12hEMA50_Volume_ATRStop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 12h Donchian channels (20-period)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Donchian channels: upper = rolling max(high, 20), lower = rolling min(low, 20)
    donch_upper = pd.Series(df_12h['high']).rolling(window=20, min_periods=20).max().values
    donch_lower = pd.Series(df_12h['low']).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 4h timeframe (completed 12h bar only)
    donch_upper_aligned = align_htf_to_ltf(prices, df_12h, donch_upper)
    donch_lower_aligned = align_htf_to_ltf(prices, df_12h, donch_lower)
    
    # 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Volume confirmation (1.8x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_bar = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Start after warmup (need enough for EMA50 and ATR calculations)
    start_idx = 60  # EMA50 needs 50 bars, plus buffer
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(donch_upper_aligned[i]) or np.isnan(donch_lower_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_spike[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above Donchian upper with 1d EMA50 uptrend and volume spike
            if close[i] > donch_upper_aligned[i] and ema_50_aligned[i] > ema_50_aligned[i-1] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
                entry_bar = i
                highest_since_entry = high[i]
            # Short entry: price breaks below Donchian lower with 1d EMA50 downtrend and volume spike
            elif close[i] < donch_lower_aligned[i] and ema_50_aligned[i] < ema_50_aligned[i-1] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
                entry_bar = i
                lowest_since_entry = low[i]
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Update highest high since entry
            highest_since_entry = max(highest_since_entry, high[i])
            # ATR trailing stop: exit when price < highest_high_since_entry - 2.0 * ATR
            if close[i] < highest_since_entry - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Update lowest low since entry
            lowest_since_entry = min(lowest_since_entry, low[i])
            # ATR trailing stop: exit when price > lowest_low_since_entry + 2.0 * ATR
            if close[i] > lowest_since_entry + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals