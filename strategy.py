#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Power with 1w trend filter and volume confirmation
# - Bull Power = High - EMA13(1w), Bear Power = EMA13(1w) - Low
# - Long when Bull Power > 0 AND Bear Power < 0 AND 1w EMA50 rising AND volume > 1.5x 20-bar avg
# - Short when Bear Power > 0 AND Bull Power < 0 AND 1w EMA50 falling AND volume > 1.5x 20-bar avg
# - Exit when power signals weaken (Bull Power < 0 for longs, Bear Power < 0 for shorts)
# - Uses 1w EMA50 for strong trend filter to avoid whipsaws in ranging markets
# - Discrete position sizing (0.25) to minimize fee churn
# - Target: 12-25 trades/year on 6h timeframe (50-100 total over 4 years)
# - Elder Ray measures bull/bear strength relative to EMA, effective in both trending and ranging markets

name = "6h_1w_elder_ray_power_v1"
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
    
    # Pre-compute 1w EMA(13) for Elder Ray calculation
    close_1w = df_1w['close'].values
    ema13_1w = pd.Series(close_1w).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Pre-compute 1w EMA(50) for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF indicators to LTF
    ema13_1w_aligned = align_htf_to_ltf(prices, df_1w, ema13_1w)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Pre-compute volume confirmation: > 1.5x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (1.5 * volume_20_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(ema13_1w_aligned[i]) or np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(volume_20_avg[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Calculate Elder Ray Power for current bar
        bull_power = prices['high'].iloc[i] - ema13_1w_aligned[i]
        bear_power = ema13_1w_aligned[i] - prices['low'].iloc[i]
        
        if position == 0:  # Flat - look for new entries
            # Long when Bull Power > 0 AND Bear Power < 0 (bulls in control)
            # AND 1w uptrend with volume spike
            if (bull_power > 0 and bear_power < 0 and 
                prices['close'].iloc[i] > ema50_1w_aligned[i] and  # price above 1w EMA50
                vol_spike.iloc[i]):
                position = 1
                signals[i] = 0.25
            # Short when Bear Power > 0 AND Bull Power < 0 (bears in control)
            # AND 1w downtrend with volume spike
            elif (bear_power > 0 and bull_power < 0 and 
                  prices['close'].iloc[i] < ema50_1w_aligned[i] and  # price below 1w EMA50
                  vol_spike.iloc[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit when power weakens
            # Exit long when Bull Power <= 0 (bulls losing control)
            # Exit short when Bear Power <= 0 (bears losing control)
            exit_signal = False
            if position == 1:  # Long position
                if bull_power <= 0:
                    exit_signal = True
            elif position == -1:  # Short position
                if bear_power <= 0:
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