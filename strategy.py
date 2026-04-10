#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w volume spike filter
# - Donchian(20) identifies breakouts from 20-day price channels
# - 1w volume spike (>1.8x 10-week average volume) confirms institutional participation
# - ATR(10) trailing stop (1.5x) adapts to volatility and manages risk
# - Discrete position sizing (0.25) minimizes fee churn
# - Target: 15-25 trades/year (60-100 total over 4 years) to avoid fee drag
# - Works in both bull and bear markets by capturing breakouts with volume confirmation

name = "1d_1w_donchian_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Pre-compute 1w volume and its moving average
    volume_1w = df_1w['volume'].values
    volume_ma_10_1w = pd.Series(volume_1w).rolling(window=10, min_periods=10).mean().values
    volume_ma_aligned = align_htf_to_ltf(prices, df_1w, volume_ma_10_1w)
    
    # Pre-compute 1d Donchian channels (20-period)
    high_1d = prices['high'].values
    low_1d = prices['low'].values
    close_1d = prices['close'].values
    
    highest_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Pre-compute 1d ATR for trailing stop (10-period)
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = np.nan
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum.reduce([tr1, tr2, tr3])
    atr = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    highest_since_entry = 0.0  # for trailing stop
    lowest_since_entry = 0.0   # for trailing stop
    
    for i in range(30, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(atr[i]) or np.isnan(volume_ma_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Get current 1w volume for filter (use raw volume, not ATR-normalized)
        volume_1w_current = volume_1w
        volume_1w_aligned = align_htf_to_ltf(prices, df_1w, volume_1w_current)
        
        # Volume confirmation: current 1w volume > 1.8x 10-week average
        volume_confirm = volume_1w_aligned[i] > 1.8 * volume_ma_aligned[i]
        
        close_price = close_1d[i]
        highest_high_val = highest_high[i]
        lowest_low_val = lowest_low[i]
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: price breaks above upper Donchian AND volume confirmation
            if close_price > highest_high_val and volume_confirm:
                position = 1
                entry_price = prices['open'].iloc[i+1] if i+1 < n else prices['close'].iloc[i]
                highest_since_entry = prices['high'].iloc[i]
                signals[i] = 0.25
            # Short conditions: price breaks below lower Donchian AND volume confirmation
            elif close_price < lowest_low_val and volume_confirm:
                position = -1
                entry_price = prices['open'].iloc[i+1] if i+1 < n else prices['close'].iloc[i]
                lowest_since_entry = prices['low'].iloc[i]
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit or trailing stop
            # Update highest/lowest since entry for trailing stop
            if position == 1:
                highest_since_entry = max(highest_since_entry, prices['high'].iloc[i])
                # ATR trailing stop: exit when price drops 1.5*ATR from highest point
                trailing_stop = prices['close'].iloc[i] < highest_since_entry - 1.5 * atr[i]
                exit_condition = trailing_stop
            else:  # position == -1
                lowest_since_entry = min(lowest_since_entry, prices['low'].iloc[i])
                # ATR trailing stop: exit when price rises 1.5*ATR from lowest point
                trailing_stop = prices['close'].iloc[i] > lowest_since_entry + 1.5 * atr[i]
                exit_condition = trailing_stop
            
            if exit_condition:
                position = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals