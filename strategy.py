#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R + 12h EMA trend filter + volume confirmation
# - Primary signal: 6h Williams %R(14) < -80 for long, > -20 for short (oversold/overbought)
# - Trend filter: only trade long when 12h EMA50 > EMA200 (bullish bias), short when EMA50 < EMA200 (bearish bias)
# - Volume confirmation: 6h volume > 20-period median volume to avoid low-participation signals
# - Position size: 0.25 (discrete level) to minimize fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) per 6h strategy guidelines
# - Works in bull/bear: Williams %R captures mean reversion swings, EMA filter ensures trend alignment

name = "6h_12h_williams_r_ema_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Pre-compute 12h indicators
    close_12h = df_12h['close'].values
    
    # 12h EMA50 and EMA200 for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200_12h = pd.Series(close_12h).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align 12h EMAs to 6h timeframe (completed 12h bar only)
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    ema_200_aligned = align_htf_to_ltf(prices, df_12h, ema_200_12h)
    
    # 6h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 6h Williams %R(14)
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close) / (highest_high_14 - lowest_low_14)
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high_14 - lowest_low_14) == 0, -50, williams_r)
    
    # 6h volume regime: volume > 20-period median volume
    median_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    volume_regime = volume > median_volume_20
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_aligned[i]) or
            np.isnan(ema_200_aligned[i]) or
            np.isnan(williams_r[i]) or
            np.isnan(volume_regime[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Williams %R crosses above -50 (mean reversion complete) OR
            #         trend turns bearish (EMA50 < EMA200)
            if williams_r[i] > -50 or ema_50_aligned[i] < ema_200_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Williams %R crosses below -50 (mean reversion complete) OR
            #         trend turns bullish (EMA50 > EMA200)
            if williams_r[i] < -50 or ema_50_aligned[i] > ema_200_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Williams %R extreme with volume confirmation and trend alignment
            # Long: Williams %R < -80 (oversold) AND volume regime AND bullish trend (EMA50 > EMA200)
            if williams_r[i] < -80 and volume_regime[i] and ema_50_aligned[i] > ema_200_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Short: Williams %R > -20 (overbought) AND volume regime AND bearish trend (EMA50 < EMA200)
            elif williams_r[i] > -20 and volume_regime[i] and ema_50_aligned[i] < ema_200_aligned[i]:
                position = -1
                signals[i] = -0.25
    
    return signals