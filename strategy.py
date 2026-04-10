#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d volume confirmation and chop filter
# - Long when price breaks above Donchian(20) high AND volume > 1.5x 20-bar avg AND chop < 61.8 (trending)
# - Short when price breaks below Donchian(20) low AND volume > 1.5x 20-bar avg AND chop < 61.8 (trending)
# - Exit with ATR trailing stop: signal=0 when long and price < highest_high - 2.5*ATR, or short and price > lowest_low + 2.5*ATR
# - Uses 1d volume for confirmation to avoid false breakouts
# - Chop filter ensures we only trade in trending regimes, avoiding whipsaws in ranges
# - Discrete position sizing (0.25) to minimize fee churn
# - Target: 25-40 trades/year on 4h timeframe (100-160 total over 4 years)

name = "4h_1d_donchian_breakout_volume_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute Donchian channels (20-period) from 4h data
    high_20 = prices['high'].rolling(window=20, min_periods=20).max().values
    low_20 = prices['low'].rolling(window=20, min_periods=20).min().values
    
    # Pre-compute ATR(14) for trailing stop
    tr1 = prices['high'] - prices['low']
    tr2 = abs(prices['high'] - prices['close'].shift(1))
    tr3 = abs(prices['low'] - prices['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # Pre-compute 1d volume confirmation: > 1.5x 20-period average
    volume_1d = df_1d['volume'].values
    volume_20_avg_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = volume_1d > (1.5 * volume_20_avg_1d)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    
    # Pre-compute Chopiness Index(14) for regime filter
    # Chop = 100 * log10(sum(ATR(1),14) / (log10(highest_high - lowest_low,14) * sqrt(14)))
    atr1 = tr  # ATR(1) is just true range
    sum_atr1 = atr1.rolling(window=14, min_periods=14).sum().values
    highest_high = prices['high'].rolling(window=14, min_periods=14).max().values
    lowest_low = prices['low'].rolling(window=14, min_periods=14).min().values
    highest_lowest_diff = highest_high - lowest_low
    # Avoid division by zero
    chop_raw = 100 * np.log10(sum_atr1 / (np.log10(highest_lowest_diff) * np.sqrt(14)))
    chop = np.where((highest_lowest_diff > 0) & (sum_atr1 > 0), chop_raw, 50.0)  # default to neutral
    chop_filter = chop < 61.8  # trending regime
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    highest_high_since_entry = 0.0
    lowest_low_since_entry = 0.0
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(atr[i]) or np.isnan(vol_spike_1d_aligned[i]) or 
            np.isnan(chop_filter[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new breakout entries
            # Long when price breaks above Donchian high AND volume spike AND trending regime
            if (prices['close'].iloc[i] > high_20[i] and 
                vol_spike_1d_aligned[i] and 
                chop_filter[i]):
                position = 1
                highest_high_since_entry = prices['high'].iloc[i]
                lowest_low_since_entry = prices['low'].iloc[i]
                signals[i] = 0.25
            # Short when price breaks below Donchian low AND volume spike AND trending regime
            elif (prices['close'].iloc[i] < low_20[i] and 
                  vol_spike_1d_aligned[i] and 
                  chop_filter[i]):
                position = -1
                highest_high_since_entry = prices['high'].iloc[i]
                lowest_low_since_entry = prices['low'].iloc[i]
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for ATR trailing stop exit
            # Update highest/lowest since entry
            if position == 1:
                highest_high_since_entry = max(highest_high_since_entry, prices['high'].iloc[i])
                lowest_low_since_entry = min(lowest_low_since_entry, prices['low'].iloc[i])
                # Exit long when price drops below highest_high - 2.5*ATR
                if prices['close'].iloc[i] < (highest_high_since_entry - 2.5 * atr[i]):
                    position = 0
                    highest_high_since_entry = 0.0
                    lowest_low_since_entry = 0.0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            elif position == -1:
                highest_high_since_entry = max(highest_high_since_entry, prices['high'].iloc[i])
                lowest_low_since_entry = min(lowest_low_since_entry, prices['low'].iloc[i])
                # Exit short when price rises above lowest_low + 2.5*ATR
                if prices['close'].iloc[i] > (lowest_low_since_entry + 2.5 * atr[i]):
                    position = 0
                    highest_high_since_entry = 0.0
                    lowest_low_since_entry = 0.0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals