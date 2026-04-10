#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R mean reversion with 1d volume confirmation and 1w trend filter
# - Long when Williams %R < -80 (oversold) in 1w uptrend (close > EMA50) with volume spike
# - Short when Williams %R > -20 (overbought) in 1w downtrend (close < EMA50) with volume spike
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Targets 12-37 trades/year (50-150 total over 4 years) to avoid fee drag
# - Weekly trend filter reduces false signals in ranging markets
# - Williams %R calculated on 12h data for precise entry timing

name = "12h_1w_williamsr_meanrev_volume_trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    if len(df_1w) < 50 or len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 1w indicators
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # 1w EMA(50) for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # 1w volume confirmation: > 1.8x 20-period average
    avg_volume_20_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    vol_spike_1w = volume_1w > (1.8 * avg_volume_20_1w)
    vol_spike_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_spike_1w)
    
    # Pre-compute Williams %R on 12h data (primary timeframe)
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    close_12h = prices['close'].values
    
    # Williams %R(14) = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close_12h) / (highest_high - lowest_low) * -100
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_spike_1w_aligned[i]) or 
            np.isnan(williams_r[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long signal: Williams %R oversold (< -80) in 1w uptrend with volume spike
            if (williams_r[i] < -80 and 
                prices['close'].iloc[i] > ema_50_1w_aligned[i] and 
                vol_spike_1w_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short signal: Williams %R overbought (> -20) in 1w downtrend with volume spike
            elif (williams_r[i] > -20 and 
                  prices['close'].iloc[i] < ema_50_1w_aligned[i] and 
                  vol_spike_1w_aligned[i]):
                position = -1
                signals[i] = -0.25
        elif position == 1:  # Long position - exit when Williams %R reaches overbought
            if williams_r[i] > -20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position - exit when Williams %R reaches oversold
            if williams_r[i] < -80:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
    
    return signals