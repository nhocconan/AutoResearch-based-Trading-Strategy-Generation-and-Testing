#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R mean reversion with 1d trend filter and volume confirmation
# - Long when Williams %R(14) crosses above -80 (oversold) in 1d uptrend (close > EMA50) with volume spike
# - Short when Williams %R(14) crosses below -20 (overbought) in 1d downtrend (close < EMA50) with volume spike
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Targets 12-37 trades/year (50-150 total over 4 years) to avoid fee drag
# - Daily trend filter reduces false signals in ranging markets
# - Williams %R provides mean reversion signals that work in both bull and bear markets

name = "12h_1d_williamsr_meanrev_volume_trend_v1"
timeframe = "12h"
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
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 1d Williams %R(14)
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r_1d = -100 * (highest_high_14 - close_1d) / (highest_high_14 - lowest_low_14 + 1e-10)
    williams_r_1d_aligned = align_htf_to_ltf(prices, df_1d, williams_r_1d)
    
    # 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 1d volume confirmation: > 1.8x 20-period average
    avg_volume_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = volume_1d > (1.8 * avg_volume_20_1d)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(williams_r_1d_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_spike_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new signals
            # Long signal: Williams %R crosses above -80 (oversold) in 1d uptrend with volume spike
            if (williams_r_1d_aligned[i] > -80 and williams_r_1d_aligned[i-1] <= -80 and
                close_1d[min(i, len(close_1d)-1)] > ema_50_1d_aligned[i] and 
                vol_spike_1d_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short signal: Williams %R crosses below -20 (overbought) in 1d downtrend with volume spike
            elif (williams_r_1d_aligned[i] < -20 and williams_r_1d_aligned[i-1] >= -20 and
                  close_1d[min(i, len(close_1d)-1)] < ema_50_1d_aligned[i] and 
                  vol_spike_1d_aligned[i]):
                position = -1
                signals[i] = -0.25
        elif position == 1:  # Long position - exit when Williams %R crosses above -20 (overbought)
            if williams_r_1d_aligned[i] >= -20 and williams_r_1d_aligned[i-1] < -20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position - exit when Williams %R crosses below -80 (oversold)
            if williams_r_1d_aligned[i] <= -80 and williams_r_1d_aligned[i-1] > -80:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
    
    return signals