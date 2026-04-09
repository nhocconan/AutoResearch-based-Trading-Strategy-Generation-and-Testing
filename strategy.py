#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) + 1w EMA200 trend filter + volume confirmation
# - Primary signal: Elder Ray Bull Power (High - EMA13) > 0 for long, Bear Power (EMA13 - Low) > 0 for short
# - Trend filter: 1w EMA200 - ensures alignment with higher timeframe trend (works in bull/bear)
# - Volume confirmation: 6h volume > 20-period median volume (avoid low-participation signals)
# - Position size: 0.25 (discrete level) to minimize fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) per 6h strategy guidelines
# - Works in bull/bear: Elder Ray measures power behind moves, 1w EMA200 filter avoids counter-trend trades

name = "6h_1w_elderray_ema_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    # Pre-compute 1w EMA200 for trend filter
    close_1w = df_1w['close'].values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Pre-compute 6h EMA13 for Elder Ray calculation
    close_6h = prices['close'].values
    ema_13_6h = pd.Series(close_6h).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    bull_power = high_6h - ema_13_6h  # Bull Power: High - EMA13
    bear_power = ema_13_6h - low_6h   # Bear Power: EMA13 - Low
    
    # 6h volume regime: volume > 20-period median volume
    volume = prices['volume'].values
    median_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    volume_regime = volume > median_volume_20
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_200_aligned[i]) or
            np.isnan(bull_power[i]) or
            np.isnan(bear_power[i]) or
            np.isnan(volume_regime[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Bull Power <= 0 (weakening bullish momentum) OR price crosses below 1w EMA200
            if bull_power[i] <= 0 or close_6h[i] < ema_200_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Bear Power <= 0 (weakening bearish momentum) OR price crosses above 1w EMA200
            if bear_power[i] <= 0 or close_6h[i] > ema_200_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Elder Ray signals with volume confirmation and 1w EMA200 filter
            # Long: Bull Power > 0 (bullish momentum) AND volume regime AND price above 1w EMA200
            if (bull_power[i] > 0 and 
                volume_regime[i] and 
                close_6h[i] > ema_200_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short: Bear Power > 0 (bearish momentum) AND volume regime AND price below 1w EMA200
            elif (bear_power[i] > 0 and 
                  volume_regime[i] and 
                  close_6h[i] < ema_200_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals