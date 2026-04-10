#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator trend + 1d volume spike + chop regime filter
# - Williams Alligator: Jaw(13,8), Teeth(8,5), Lips(5,3) SMAs of median price
# - Trend up: Lips > Teeth > Jaw; Trend down: Lips < Teeth < Jaw
# - Enter long in Alligator uptrend + volume spike + chop < 61.8 (trending regime)
# - Enter short in Alligator downtrend + volume spike + chop < 61.8
# - Exit when Alligator reverses (Lips crosses Teeth) or chop > 61.8 (range regime)
# - Uses discrete position sizing (0.25) to balance return and fee drag
# - Alligator catches sustained trends; volume confirms institutional participation
# - Chop filter avoids whipsaws in ranging markets (critical for 2025 bear/range)
# - Target: 12-37 trades/year on 12h timeframe (50-150 total over 4 years)

name = "12h_1d_alligator_volume_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute Williams Alligator on 12h data
    median_price = (prices['high'].values + prices['low'].values) / 2.0
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().shift(8)
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().shift(5)
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().shift(3)
    
    # Alligator trends: Lips > Teeth > Jaw (up), Lips < Teeth < Jaw (down)
    alligator_up = (lips > teeth) & (teeth > jaw)
    alligator_down = (lips < teeth) & (teeth < jaw)
    
    # Pre-compute 1d volume confirmation: > 1.8x 20-period average
    volume_1d = df_1d['volume'].values
    volume_20_avg_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = volume_1d > (1.8 * volume_20_avg_1d)
    
    # Align HTF indicators to 12h timeframe
    alligator_up_aligned = align_htf_to_ltf(prices, df_1d, alligator_up.values if hasattr(alligator_up, 'values') else alligator_up)
    alligator_down_aligned = align_htf_to_ltf(prices, df_1d, alligator_down.values if hasattr(alligator_down, 'values') else alligator_down)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    
    # Pre-compute Choppiness Index on 12h data (regime filter)
    atr_period = 14
    high_low = prices['high'].values - prices['low'].values
    high_close = np.abs(prices['high'].values - np.roll(prices['close'].values, 1))
    low_close = np.abs(prices['low'].values - np.roll(prices['close'].values, 1))
    high_close[0] = high_low[0]
    low_close[0] = high_low[0]
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    hh = pd.Series(prices['high'].values).rolling(window=atr_period, min_periods=atr_period).max().values
    ll = pd.Series(prices['low'].values).rolling(window=atr_period, min_periods=atr_period).min().values
    
    # Avoid division by zero
    sum_atr = pd.Series(atr).rolling(window=atr_period, min_periods=atr_period).sum().values
    range_hl = hh - ll
    chop = np.where(range_hl != 0, 100 * np.log10(sum_atr / range_hl) / np.log10(atr_period), 50)
    
    # Chop regime: < 61.8 = trending (good for Alligator), > 61.8 = ranging (avoid)
    chop_trending = chop < 61.8
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(alligator_up_aligned[i]) or np.isnan(alligator_down_aligned[i]) or
            np.isnan(vol_spike_1d_aligned[i]) or np.isnan(chop_trending[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new Alligator entries
            # Long when Alligator uptrend + volume spike + chop trending
            if (alligator_up_aligned[i] and 
                vol_spike_1d_aligned[i] and 
                chop_trending[i]):
                position = 1
                signals[i] = 0.25
            # Short when Alligator downtrend + volume spike + chop trending
            elif (alligator_down_aligned[i] and 
                  vol_spike_1d_aligned[i] and 
                  chop_trending[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit conditions
            # Exit when Alligator reverses OR chop enters ranging regime
            alligator_reverse = (position == 1 and not alligator_up_aligned[i]) or \
                               (position == -1 and not alligator_down_aligned[i])
            chop_range = not chop_trending[i]
            
            if alligator_reverse or chop_range:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals