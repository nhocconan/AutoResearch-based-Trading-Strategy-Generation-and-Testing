#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 1w trend filter + volume spike
# - Elder Ray: Bull Power = High - EMA13, Bear Power = EMA13 - Low
# - Long when Bull Power > 0 AND Bear Power < 0 (bullish momentum) AND volume > 1.5x 20-bar avg AND 1w close > 1w EMA21
# - Short when Bear Power > 0 AND Bull Power < 0 (bearish momentum) AND volume > 1.5x 20-bar avg AND 1w close < 1w EMA21
# - Exit when momentum reverses (Bull Power < 0 for long, Bear Power < 0 for short)
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Targets ~20-40 trades/year (80-160 total over 4 years) to avoid fee drag
# - Elder Ray captures institutional buying/selling pressure
# - Weekly trend filter ensures alignment with higher timeframe
# - Volume confirmation filters weak breakouts

name = "6h_1w_elder_ray_volume_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Pre-compute weekly EMA(21) for trend filter
    close_1w = df_1w['close'].values
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # Pre-compute 6h EMA(13) for Elder Ray calculation
    close_6h = prices['close'].values
    ema_13_6h = pd.Series(close_6h).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Pre-compute 6h Elder Ray components
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    bull_power = high_6h - ema_13_6h      # Bull Power = High - EMA13
    bear_power = ema_13_6h - low_6h       # Bear Power = EMA13 - Low
    
    # Pre-compute 6h volume confirmation: > 1.5x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (1.5 * volume_20_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema_21_1w_aligned[i]) or np.isnan(volume_20_avg[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long signal: Bull Power > 0 AND Bear Power < 0 (bullish momentum) with volume spike and weekly uptrend
            if (bull_power[i] > 0 and bear_power[i] < 0 and 
                vol_spike.iloc[i] and 
                prices['close'].iloc[i] > ema_21_1w_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short signal: Bear Power > 0 AND Bull Power < 0 (bearish momentum) with volume spike and weekly downtrend
            elif (bear_power[i] > 0 and bull_power[i] < 0 and 
                  vol_spike.iloc[i] and 
                  prices['close'].iloc[i] < ema_21_1w_aligned[i]):
                position = -1
                signals[i] = -0.25
        else:  # Have position - look for exit
            # Exit long when Bull Power turns negative (momentum fading)
            if position == 1 and bull_power[i] < 0:
                position = 0
                signals[i] = 0.0
            # Exit short when Bear Power turns negative (momentum fading)
            elif position == -1 and bear_power[i] < 0:
                position = 0
                signals[i] = 0.0
            # Hold position otherwise
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals