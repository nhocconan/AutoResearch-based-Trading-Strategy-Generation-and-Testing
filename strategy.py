#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using weekly Donchian(20) breakout with 1w EMA34 trend filter and volume spike confirmation.
# Long when: price breaks above weekly Donchian upper channel AND close > 1w EMA34 AND volume > 2.0x 20-bar average
# Short when: price breaks below weekly Donchian lower channel AND close < 1w EMA34 AND volume > 2.0x 20-bar average
# Exit via ATR(20) trailing stop: long exit when price < highest_high_since_entry - 2.5 * ATR
#                      short exit when price > lowest_low_since_entry + 2.5 * ATR
# Uses weekly Donchian for structure (proven edge on higher timeframes), 1w EMA34 for trend alignment, volume spike for confirmation
# Discrete sizing 0.28 balances return and fee drag. Target: 30-100 total trades over 4 years = 7-25/year.

name = "1d_WeeklyDonchian20_1wEMA34_VolumeSpike_ATRStop_v1"
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
    
    # Calculate weekly Donchian channels (based on previous weekly bar)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Donchian channels for each weekly bar (using previous 20 weekly bars)
    donchian_upper = np.full(len(close_1w), np.nan)
    donchian_lower = np.full(len(close_1w), np.nan)
    
    for i in range(20, len(close_1w)):
        # Donchian upper: highest high of previous 20 weekly bars
        donchian_upper[i] = np.max(high_1w[i-20:i])
        # Donchian lower: lowest low of previous 20 weekly bars
        donchian_lower[i] = np.min(low_1w[i-20:i])
    
    # Align weekly Donchian channels to daily timeframe (completed weekly bar only)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1w, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1w, donchian_lower)
    
    # Calculate 1w EMA34 for trend filter
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Daily ATR(20) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Volume confirmation (2.0x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_bar = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Start after warmup (need enough for ATR, Donchian, EMA calculations)
    start_idx = 20 + 34 + 5  # ATR(20) + EMA34 warmup + buffer
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(volume_spike[i]) or 
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above weekly Donchian upper with volume spike AND bullish trend (close > 1w EMA34)
            if close[i] > donchian_upper_aligned[i] and volume_spike[i] and close[i] > ema_34_1w_aligned[i]:
                signals[i] = 0.28
                position = 1
                entry_bar = i
                highest_since_entry = high[i]
            # Short entry: price breaks below weekly Donchian lower with volume spike AND bearish trend (close < 1w EMA34)
            elif close[i] < donchian_lower_aligned[i] and volume_spike[i] and close[i] < ema_34_1w_aligned[i]:
                signals[i] = -0.28
                position = -1
                entry_bar = i
                lowest_since_entry = low[i]
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Update highest high since entry
            highest_since_entry = max(highest_since_entry, high[i])
            # ATR trailing stop: exit when price < highest_high_since_entry - 2.5 * ATR
            if close[i] < highest_since_entry - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.28
        
        elif position == -1:  # Short position
            # Update lowest low since entry
            lowest_since_entry = min(lowest_since_entry, low[i])
            # ATR trailing stop: exit when price > lowest_low_since_entry + 2.5 * ATR
            if close[i] > lowest_since_entry + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.28
    
    return signals