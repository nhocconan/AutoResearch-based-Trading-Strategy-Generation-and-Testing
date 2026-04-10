#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 12h trend filter + volume confirmation
# - Elder Ray: Bull Power = High - EMA13, Bear Power = EMA13 - Low (12h timeframe)
# - Long when Bull Power > 0 AND Bear Power < 0 (bullish momentum) AND volume > 1.5x 20-bar average AND 12h close > 12h EMA50
# - Short when Bear Power > 0 AND Bull Power < 0 (bearish momentum) AND volume > 1.5x 20-bar average AND 12h close < 12h EMA50
# - Exit when momentum reverses (Bull Power < 0 for long, Bear Power < 0 for short) OR volume drops below 0.7x average
# - Uses 12h EMA50 trend filter to avoid counter-trend trades
# - Elder Ray captures institutional buying/selling pressure - works in both bull and bear markets
# - Moderate volume threshold (1.5x) and momentum-based exits target 15-25 trades/year
# - Focus on BTC/ETH; SOL-only strategies are low value

name = "6h_12h_elder_ray_volume_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - 12h for Elder Ray and trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Pre-compute volume confirmation: > 1.5x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (1.5 * volume_20_avg)
    vol_weak = prices['volume'] < (0.7 * volume_20_avg)
    
    # Pre-compute 12h data arrays
    h_12h = df_12h['high'].values
    l_12h = df_12h['low'].values
    c_12h = df_12h['close'].values
    
    # Align 12h data to 6h timeframe (each 12h bar = 2x 6h bars)
    h_12h_aligned = align_htf_to_ltf(prices, df_12h, h_12h)
    l_12h_aligned = align_htf_to_ltf(prices, df_12h, l_12h)
    c_12h_aligned = align_htf_to_ltf(prices, df_12h, c_12h)
    
    # Pre-compute 12h EMA(13) for Elder Ray
    ema13_12h = pd.Series(c_12h).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema13_12h_aligned = align_htf_to_ltf(prices, df_12h, ema13_12h)
    
    # Pre-compute 12h EMA(50) for trend filter
    ema50_12h = pd.Series(c_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(ema13_12h_aligned[i]) or np.isnan(ema50_12h_aligned[i]) or 
            np.isnan(volume_20_avg[i]) or np.isnan(h_12h_aligned[i]) or 
            np.isnan(l_12h_aligned[i]) or np.isnan(c_12h_aligned[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Calculate Elder Ray components for current bar
        bull_power = h_12h_aligned[i] - ema13_12h_aligned[i]  # High - EMA13
        bear_power = ema13_12h_aligned[i] - l_12h_aligned[i]  # EMA13 - Low
        
        # Get previous bar values for momentum check (to avoid whipsaws)
        if i >= 1:
            prev_bull_power = h_12h_aligned[i-1] - ema13_12h_aligned[i-1]
            prev_bear_power = ema13_12h_aligned[i-1] - l_12h_aligned[i-1]
        else:
            prev_bull_power = bull_power
            prev_bear_power = bear_power
        
        if position == 0:  # Flat - look for new entries
            # Long entry: bullish momentum (Bull Power > 0 AND Bear Power < 0) with volume spike AND uptrend
            if (bull_power > 0 and bear_power < 0 and 
                vol_spike.iloc[i] and 
                c_12h_aligned[i] > ema50_12h_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: bearish momentum (Bear Power > 0 AND Bull Power < 0) with volume spike AND downtrend
            elif (bear_power > 0 and bull_power < 0 and 
                  vol_spike.iloc[i] and 
                  c_12h_aligned[i] < ema50_12h_aligned[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0  # Stay flat
        elif position == 1:  # Long position - look for exit
            # Exit conditions:
            # 1. Momentum reverses (Bull Power < 0 indicates losing bullish momentum)
            # 2. Volume drops below 0.7x average (loss of momentum)
            if (bull_power < 0 or vol_weak.iloc[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Hold long
        elif position == -1:  # Short position - look for exit
            # Exit conditions:
            # 1. Momentum reverses (Bear Power < 0 indicates losing bearish momentum)
            # 2. Volume drops below 0.7x average (loss of momentum)
            if (bear_power < 0 or vol_weak.iloc[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Hold short
    
    return signals