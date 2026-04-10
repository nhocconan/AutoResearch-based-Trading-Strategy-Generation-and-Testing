#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout + 1w volume confirmation + ATR-based trailing stop
# - Long when price breaks above 20-day high AND 1w volume > 1.5x 20-period average
# - Short when price breaks below 20-day low AND 1w volume > 1.5x 20-period average
# - Exit long when price drops below 20-day ATR trailing stop from highest high since entry
# - Exit short when price rises above 20-day ATR trailing stop from lowest low since entry
# - Uses discrete position sizing 0.25 to limit fee churn
# - Target: 7-25 trades/year on 1d timeframe (30-100 total over 4 years)
# - Donchian breakouts capture strong momentum moves in both bull and bear markets
# - 1w volume confirmation ensures breakouts have institutional participation
# - ATR trailing stop manages risk and lets winners run

name = "1d_1w_donchian_volume_atrstop_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute 1d Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Pre-compute 1d ATR(20) for trailing stop
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - np.roll(close, 1)[1:])
    tr3 = np.abs(low[1:] - np.roll(close, 1)[1:])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # first element is NaN
    atr = pd.Series(tr).ewm(alpha=1/20, adjust=False, min_periods=20).mean().values
    
    # Pre-compute 1w volume confirmation
    vol_1w = df_1w['volume'].values
    vol_ma_1w = pd.Series(vol_1w).rolling(window=20, min_periods=20).mean().values
    volume_spike_1w = vol_1w > (1.5 * vol_ma_1w)
    
    # Align HTF indicators to 1d timeframe
    volume_spike_1w_aligned = align_htf_to_ltf(prices, df_1w, volume_spike_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    highest_since_entry = 0.0  # Track highest high since long entry
    lowest_since_entry = 0.0   # Track lowest low since short entry
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(atr[i]) or np.isnan(volume_spike_1w_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: price breaks above 20-day high AND 1w volume spike
            if (close[i] > highest_high[i] and 
                volume_spike_1w_aligned[i]):
                position = 1
                highest_since_entry = close[i]
                signals[i] = 0.25
            # Short conditions: price breaks below 20-day low AND 1w volume spike
            elif (close[i] < lowest_low[i] and 
                  volume_spike_1w_aligned[i]):
                position = -1
                lowest_since_entry = close[i]
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        elif position == 1:  # Long position - trail stop and check for exit
            # Update highest high since entry
            highest_since_entry = max(highest_since_entry, high[i])
            # Calculate trailing stop: highest high since entry minus 2.0 * ATR
            trailing_stop = highest_since_entry - (2.0 * atr[i])
            # Exit if price drops below trailing stop
            if close[i] < trailing_stop:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        else:  # position == -1, Short position - trail stop and check for exit
            # Update lowest low since entry
            lowest_since_entry = min(lowest_since_entry, low[i])
            # Calculate trailing stop: lowest low since entry plus 2.0 * ATR
            trailing_stop = lowest_since_entry + (2.0 * atr[i])
            # Exit if price rises above trailing stop
            if close[i] > trailing_stop:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
    
    return signals