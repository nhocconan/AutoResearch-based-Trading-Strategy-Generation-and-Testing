#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with 1d trend filter and volume confirmation
# - Long when Williams %R(14) < -80 (oversold) in 1d uptrend (close > EMA50) with volume > 1.8x 20-bar avg
# - Short when Williams %R(14) > -20 (overbought) in 1d downtrend (close < EMA50) with volume spike
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Targets ~25 trades/year (100 total over 4 years) to avoid fee drag
# - 1d trend filter reduces false signals in counter-trend markets
# - Williams %R provides timely mean reversion entries in 6h timeframe
# - Volume confirmation ensures institutional participation

name = "6h_1d_williamsr_meanrev_volume_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 1d indicators
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 1d volume confirmation: > 1.8x 20-period average
    avg_volume_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = volume_1d > (1.8 * avg_volume_20_1d)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    
    # Pre-compute Williams %R on 6h data
    highest_high_14 = prices['high'].rolling(window=14, min_periods=14).max().values
    lowest_low_14 = prices['low'].rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - prices['close'].values) / (highest_high_14 - lowest_low_14)
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high_14 - lowest_low_14) == 0, -50, williams_r)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_spike_1d_aligned[i]) or 
            np.isnan(williams_r[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long signal: Williams %R oversold in 1d uptrend with volume spike
            if (williams_r[i] < -80 and 
                prices['close'].iloc[i] > ema_50_1d_aligned[i] and 
                vol_spike_1d_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short signal: Williams %R overbought in 1d downtrend with volume spike
            elif (williams_r[i] > -20 and 
                  prices['close'].iloc[i] < ema_50_1d_aligned[i] and 
                  vol_spike_1d_aligned[i]):
                position = -1
                signals[i] = -0.25
        else:  # Have position - look for exit
            # Exit long when Williams %R returns to neutral (> -50)
            if position == 1 and williams_r[i] > -50:
                position = 0
                signals[i] = 0.0
            # Exit short when Williams %R returns to neutral (< -50)
            elif position == -1 and williams_r[i] < -50:
                position = 0
                signals[i] = 0.0
            # Hold position otherwise
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals