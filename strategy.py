#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d volume confirmation and ATR(14) stoploss
# - Long when price breaks above Donchian upper channel (20-period high) AND 1d volume > 1.3x 20-bar avg
# - Short when price breaks below Donchian lower channel (20-period low) AND 1d volume > 1.3x 20-bar avg
# - Stoploss: exit long when price < highest high since entry - 2.5 * ATR(14); exit short when price > lowest low since entry + 2.5 * ATR(14)
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Target: 20-50 trades/year on 4h timeframe (80-200 total over 4 years)
# - Donchian channels provide objective breakout levels; volume confirms institutional participation
# - ATR-based stoploss adapts to volatility, reducing whipsaws in ranging markets

name = "4h_1d_donchian_breakout_volume_atr_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Pre-compute Donchian channels (20-period) on 4h data
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    
    # Donchian upper: highest high of last 20 periods
    donchian_upper = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    # Donchian lower: lowest low of last 20 periods
    donchian_lower = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Pre-compute ATR(14) for stoploss
    tr1 = pd.Series(high_4h - low_4h)
    tr2 = pd.Series(np.abs(high_4h - np.roll(close_4h, 1)))
    tr3 = pd.Series(np.abs(low_4h - np.roll(close_4h, 1)))
    tr2.iloc[0] = tr2.iloc[1]  # Fix first value
    tr3.iloc[0] = tr3.iloc[1]  # Fix first value
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Pre-compute 1d volume confirmation: > 1.3x 20-period average
    volume_1d = df_1d['volume'].values
    volume_20_avg = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = volume_1d > (1.3 * volume_20_avg)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    highest_high_since_entry = 0.0
    lowest_low_since_entry = 0.0
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(atr[i]) or np.isnan(vol_spike_1d_aligned[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new breakout entries
            # Long when price breaks above Donchian upper AND 1d volume spike
            if (prices['close'].iloc[i] > donchian_upper[i] and 
                vol_spike_1d_aligned[i]):
                position = 1
                entry_price = prices['open'].iloc[i+1] if i+1 < n else prices['close'].iloc[i]
                highest_high_since_entry = entry_price
                lowest_low_since_entry = entry_price
                signals[i] = 0.25
            # Short when price breaks below Donchian lower AND 1d volume spike
            elif (prices['close'].iloc[i] < donchian_lower[i] and 
                  vol_spike_1d_aligned[i]):
                position = -1
                entry_price = prices['open'].iloc[i+1] if i+1 < n else prices['close'].iloc[i]
                highest_high_since_entry = entry_price
                lowest_low_since_entry = entry_price
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - update extremes and check ATR stoploss
            # Update highest high and lowest low since entry
            if position == 1:  # Long position
                highest_high_since_entry = max(highest_high_since_entry, prices['high'].iloc[i])
                lowest_low_since_entry = min(lowest_low_since_entry, prices['low'].iloc[i])
                # Stoploss: price < highest high since entry - 2.5 * ATR
                if prices['close'].iloc[i] < highest_high_since_entry - 2.5 * atr[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            elif position == -1:  # Short position
                highest_high_since_entry = max(highest_high_since_entry, prices['high'].iloc[i])
                lowest_low_since_entry = min(lowest_low_since_entry, prices['low'].iloc[i])
                # Stoploss: price > lowest low since entry + 2.5 * ATR
                if prices['close'].iloc[i] > lowest_low_since_entry + 2.5 * atr[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals