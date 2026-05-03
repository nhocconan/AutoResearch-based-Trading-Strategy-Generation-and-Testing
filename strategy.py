#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using weekly Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation
# Long when price breaks above weekly Donchian high with 1d EMA50 uptrend and volume > 1.5x 20-bar average
# Short when price breaks below weekly Donchian low with 1d EMA50 downtrend and volume > 1.5x 20-bar average
# Exit via ATR(14) trailing stop: long exit when price < highest_high_since_entry - 2.0 * ATR
#                      short exit when price > lowest_low_since_entry + 2.0 * ATR
# Uses weekly price channels for structure, daily EMA50 for trend filter, volume for confirmation
# Discrete sizing 0.25 balances return and fee drag. Target: 30-100 total trades over 4 years = 7-25/year.

name = "1d_WeeklyDonchian20_1dEMA50_Volume_ATRStop_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate weekly Donchian channels (20-period high/low)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly Donchian: highest high and lowest low over past 20 weekly bars
    dh_high = pd.Series(df_1w['high']).rolling(window=20, min_periods=20).max().values
    dh_low = pd.Series(df_1w['low']).rolling(window=20, min_periods=20).min().values
    
    # Align weekly Donchian levels to daily timeframe (completed weekly bar only)
    dh_high_aligned = align_htf_to_ltf(prices, df_1w, dh_high)
    dh_low_aligned = align_htf_to_ltf(prices, df_1w, dh_low)
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values  # Using weekly close for EMA50
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1d)
    
    # ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Volume confirmation (1.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_bar = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Start after warmup (need enough for calculations)
    start_idx = 100  # EMA50 needs 50 bars, plus buffer
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(dh_high_aligned[i]) or np.isnan(dh_low_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_spike[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above weekly Donchian high with 1d EMA50 uptrend and volume spike
            if close[i] > dh_high_aligned[i] and ema_50_aligned[i] > ema_50_aligned[i-1] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
                entry_bar = i
                highest_since_entry = high[i]
            # Short entry: price breaks below weekly Donchian low with 1d EMA50 downtrend and volume spike
            elif close[i] < dh_low_aligned[i] and ema_50_aligned[i] < ema_50_aligned[i-1] and volume_spike[i]:
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