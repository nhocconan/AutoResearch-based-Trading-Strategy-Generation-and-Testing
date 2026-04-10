#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1w trend filter and volume confirmation
# - Bull Power = High - EMA13(1w), Bear Power = EMA13(1w) - Low
# - Long when Bull Power > 0 AND Bear Power < 0 (bullish imbalance) AND volume > 1.5x average AND 1w close > 1w EMA34
# - Short when Bear Power > 0 AND Bull Power < 0 (bearish imbalance) AND volume > 1.5x average AND 1w close < 1w EMA34
# - Exit when power imbalance reverses OR volume drops below 0.7x average
# - Uses 1w trend filter to avoid counter-trend trades and focus on strong momentum
# - Elder Ray measures buying/selling pressure relative to trend (EMA13)
# - Volume confirmation filters weak breakouts
# - Target: 12-25 trades/year (50-100 total over 4 years) to minimize fee drag

name = "6h_1w_elder_ray_volume_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Pre-compute 1w EMA(13) for Elder Ray calculation
    close_1w = df_1w['close'].values
    ema13_1w = pd.Series(close_1w).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema13_1w_aligned = align_htf_to_ltf(prices, df_1w, ema13_1w)
    
    # Pre-compute 1w EMA(34) for trend filter
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Pre-compute volume confirmation: > 1.5x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (1.5 * volume_20_avg)
    
    # Pre-compute volume filter: < 0.7x average volume for exit (loss of momentum)
    vol_weak = prices['volume'] < (0.7 * volume_20_avg)
    
    # Pre-compute aligned 1w data properly
    h_1w = df_1w['high'].values
    l_1w = df_1w['low'].values
    c_1w = df_1w['close'].values
    
    # Align them to 6h timeframe
    h_1w_aligned = align_htf_to_ltf(prices, df_1w, h_1w)
    l_1w_aligned = align_htf_to_ltf(prices, df_1w, l_1w)
    c_1w_aligned = align_htf_to_ltf(prices, df_1w, c_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(ema13_1w_aligned[i]) or np.isnan(ema34_1w_aligned[i]) or 
            np.isnan(volume_20_avg[i]) or np.isnan(h_1w_aligned[i]) or 
            np.isnan(l_1w_aligned[i]) or np.isnan(c_1w_aligned[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Calculate Elder Ray components using current 6h bar's high/low and 1w EMA13
        bull_power = h_1w_aligned[i] - ema13_1w_aligned[i]  # High - EMA13
        bear_power = ema13_1w_aligned[i] - l_1w_aligned[i]  # EMA13 - Low
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Bull Power > 0 AND Bear Power < 0 (bullish imbalance) 
            #           AND volume spike AND 1w uptrend
            if (bull_power > 0 and bear_power < 0 and 
                vol_spike.iloc[i] and 
                c_1w_aligned[i] > ema34_1w_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: Bear Power > 0 AND Bull Power < 0 (bearish imbalance)
            #          AND volume spike AND 1w downtrend
            elif (bear_power > 0 and bull_power < 0 and 
                  vol_spike.iloc[i] and 
                  c_1w_aligned[i] < ema34_1w_aligned[i]):
                position = -1
                signals[i] = -0.25
        else:  # Have position - look for exit
            # Exit conditions:
            # 1. Power imbalance reverses (trend weakness)
            # 2. Volume drops below 0.7x average (loss of momentum)
            if position == 1:  # Long position
                if (bull_power <= 0 or bear_power >= 0 or vol_weak.iloc[i]):
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25  # Hold long
            elif position == -1:  # Short position
                if (bear_power <= 0 or bull_power >= 0 or vol_weak.iloc[i]):
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25  # Hold short
    
    return signals