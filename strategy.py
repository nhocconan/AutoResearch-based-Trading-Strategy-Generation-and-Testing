#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band squeeze breakout with 1d trend filter and volume confirmation
# - Long when 6h price breaks above upper BB(20,2) AND 1d close > EMA50 AND volume > 2.0x 20-bar avg
# - Short when 6h price breaks below lower BB(20,2) AND 1d close < EMA50 AND volume > 2.0x 20-bar avg
# - Exit when price returns to middle BB(20) or opposite band touch
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Targets ~20 trades/year (80 total over 4 years) to avoid fee drag
# - Bollinger squeeze identifies low volatility periods primed for breakout
# - 1d trend filter ensures alignment with higher timeframe momentum
# - Volume confirmation filters weak breakouts

name = "6h_1d_bb_squeeze_breakout_trend_volume_v1"
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
    
    # 1d volume confirmation: > 2.0x 20-period average
    avg_volume_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = volume_1d > (2.0 * avg_volume_20_1d)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    
    # Pre-compute 6h Bollinger Bands (20,2)
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Middle band = SMA(20)
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    # Standard deviation
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    # Upper and lower bands
    upper_bb = sma_20 + (2.0 * std_20)
    lower_bb = sma_20 - (2.0 * std_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_spike_1d_aligned[i]) or 
            np.isnan(sma_20[i]) or np.isnan(upper_bb[i]) or np.isnan(lower_bb[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long signal: price breaks above upper BB in 1d uptrend with volume spike
            if (close[i] > upper_bb[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                vol_spike_1d_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short signal: price breaks below lower BB in 1d downtrend with volume spike
            elif (close[i] < lower_bb[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  vol_spike_1d_aligned[i]):
                position = -1
                signals[i] = -0.25
        else:  # Have position - look for exit
            # Exit long when price returns to middle BB or touches lower BB
            if position == 1 and (close[i] <= sma_20[i] or close[i] < lower_bb[i]):
                position = 0
                signals[i] = 0.0
            # Exit short when price returns to middle BB or touches upper BB
            elif position == -1 and (close[i] >= sma_20[i] or close[i] > upper_bb[i]):
                position = 0
                signals[i] = 0.0
            # Hold position otherwise
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals