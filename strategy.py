#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Power + 1w Trend Filter
# - Elder Ray: Bull Power = High - EMA13(close), Bear Power = EMA13(close) - Low
# - Long when Bull Power > 0 AND Bear Power < 0 AND 1w EMA21 rising (bullish regime)
# - Short when Bear Power > 0 AND Bull Power < 0 AND 1w EMA21 falling (bearish regime)
# - Exit when power signs diverge (loss of momentum)
# - Uses 1w EMA21 for regime filter to avoid counter-trend trades
# - Discrete position sizing (0.25) to minimize fee churn
# - Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years)
# - Elder Ray measures bull/bear power relative to EMA; works in both regimes with trend filter

name = "6h_1w_elder_ray_power_regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:
        return np.zeros(n)
    
    # Pre-compute 13-period EMA for Elder Ray (LTF)
    close = prices['close'].values
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray Power
    bull_power = prices['high'].values - ema13  # High - EMA13
    bear_power = ema13 - prices['low'].values  # EMA13 - Low
    
    # Pre-compute 1w EMA(21) for regime filter
    close_1w = df_1w['close'].values
    ema21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema21_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(13, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema21_1w_aligned[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long when bull power positive AND bear power negative AND 1w uptrend
            if (bull_power[i] > 0 and bear_power[i] < 0 and 
                close[i] > ema21_1w_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short when bear power positive AND bull power negative AND 1w downtrend
            elif (bear_power[i] > 0 and bull_power[i] < 0 and 
                  close[i] < ema21_1w_aligned[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit (loss of momentum)
            # Exit when power signs diverge (both positive or both negative = weakening)
            exit_signal = False
            if position == 1:  # Long position
                if bull_power[i] <= 0 or bear_power[i] >= 0:  # Loss of bull power or bear power turning positive
                    exit_signal = True
            elif position == -1:  # Short position
                if bear_power[i] <= 0 or bull_power[i] >= 0:  # Loss of bear power or bull power turning positive
                    exit_signal = True
            
            if exit_signal:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals