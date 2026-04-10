#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Power + 1w Regime Filter
# - Bull Power = High - EMA13(close), Bear Power = EMA13(close) - Low
# - Long when Bull Power > 0 AND 1w EMA34 rising AND Bull Power > Bear Power
# - Short when Bear Power > 0 AND 1w EMA34 falling AND Bear Power > Bull Power
# - Exit when power diverges (Bull Power < 0 for longs, Bear Power < 0 for shorts)
# - Uses weekly EMA34 for regime filter to avoid counter-trend trades
# - Discrete position sizing (0.25) to minimize fee churn
# - Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years)
# - Elder Ray captures institutional buying/selling pressure; weekly regime avoids whipsaws

name = "6h_1w_elder_ray_power_regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Pre-compute 13-period EMA for Elder Ray (using close)
    close = prices['close'].values
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray Power
    bull_power = prices['high'].values - ema13  # High - EMA13
    bear_power = ema13 - prices['low'].values  # EMA13 - Low
    
    # Pre-compute 1w EMA34 for regime filter
    close_1w = df_1w['close'].values
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Pre-compute 1w EMA34 slope (rising/falling)
    ema34_slope = np.zeros_like(ema34_1w_aligned)
    ema34_slope[1:] = ema34_1w_aligned[1:] - ema34_1w_aligned[:-1]
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(13, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema34_1w_aligned[i]) or np.isnan(ema34_slope[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long when Bull Power > 0 AND 1w uptrend AND Bull Power > Bear Power
            if (bull_power[i] > 0 and 
                ema34_slope[i] > 0 and  # 1w EMA34 rising
                bull_power[i] > bear_power[i]):
                position = 1
                signals[i] = 0.25
            # Short when Bear Power > 0 AND 1w downtrend AND Bear Power > Bull Power
            elif (bear_power[i] > 0 and 
                  ema34_slope[i] < 0 and  # 1w EMA34 falling
                  bear_power[i] > bull_power[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit when power diverges
            # Exit when power goes negative (loss of buying/selling pressure)
            exit_signal = False
            if position == 1:  # Long position
                if bull_power[i] <= 0:
                    exit_signal = True
            elif position == -1:  # Short position
                if bear_power[i] <= 0:
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