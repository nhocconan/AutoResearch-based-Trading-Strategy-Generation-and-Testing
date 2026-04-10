#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with 1w trend filter and volume confirmation
# - Long when Williams %R(14) crosses above -80 from below in 1w uptrend (close > EMA50) with volume spike
# - Short when Williams %R(14) crosses below -20 from above in 1w downtrend (close < EMA50) with volume spike
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Targets 12-37 trades/year (50-150 total over 4 years) to avoid fee drag
# - Weekly trend filter reduces false signals in ranging markets
# - Williams %R is effective in both bull and bear markets for mean reversion entries

name = "6h_1w_williamsr_meanrev_volume_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Pre-compute 1w indicators
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    volume_1w = df_1w['volume'].values
    
    # 1w EMA(50) for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # 1w volume confirmation: > 2.0x 20-period average
    avg_volume_20_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    vol_spike_1w = volume_1w > (2.0 * avg_volume_20_1w)
    vol_spike_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_spike_1w)
    
    # Pre-compute Williams %R on 6h data
    highest_high_14 = pd.Series(prices['high']).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(prices['low']).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - prices['close'].values) / (highest_high_14 - lowest_low_14)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_spike_1w_aligned[i]) or 
            np.isnan(williams_r[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Williams %R conditions
            williams_r_prev = williams_r[i-1] if i > 0 else -100
            
            # Long signal: Williams %R crosses above -80 from below in 1w uptrend with volume spike
            if (williams_r_prev <= -80 and williams_r[i] > -80 and 
                prices['close'].iloc[i] > ema_50_1w_aligned[i] and 
                vol_spike_1w_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short signal: Williams %R crosses below -20 from above in 1w downtrend with volume spike
            elif (williams_r_prev >= -20 and williams_r[i] < -20 and 
                  prices['close'].iloc[i] < ema_50_1w_aligned[i] and 
                  vol_spike_1w_aligned[i]):
                position = -1
                signals[i] = -0.25
        else:  # Have position - look for exit signals
            if position == 1:  # Long position
                # Exit long when Williams %R crosses below -50 (mean reversion complete)
                if williams_r[i] < -50 and williams_r[i-1] >= -50:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            elif position == -1:  # Short position
                # Exit short when Williams %R crosses above -50 (mean reversion complete)
                if williams_r[i] > -50 and williams_r[i-1] <= -50:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals