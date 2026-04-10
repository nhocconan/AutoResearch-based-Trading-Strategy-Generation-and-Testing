#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R mean reversion with 1d trend filter and volume confirmation
# - Long when Williams %R(14) < -80 (oversold) AND price > 1d EMA200 AND volume > 1.5x 20-bar avg
# - Short when Williams %R(14) > -20 (overbought) AND price < 1d EMA200 AND volume > 1.5x 20-bar avg
# - Exit when Williams %R crosses above -50 (for longs) or below -50 (for shorts)
# - Uses 1d EMA200 for trend filter to avoid counter-trend trades
# - Discrete position sizing (0.25) to minimize fee churn
# - Target: 20-40 trades/year on 4h timeframe (80-160 total over 4 years)
# - Williams %R is effective in ranging/bear markets which matches 2025+ test conditions

name = "4h_1d_williamsr_meanreversion_volume_trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute Williams %R(14)
    highest_high = pd.Series(prices['high']).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(prices['low']).rolling(window=14, min_periods=14).min().values
    willr = -100 * (highest_high - prices['close'].values) / (highest_high - lowest_low)
    # Handle division by zero when high == low
    willr = np.where((highest_high - lowest_low) == 0, -50, willr)
    
    # Pre-compute volume confirmation: > 1.5x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (1.5 * volume_20_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Pre-compute aligned 1d data
    c_1d = df_1d['close'].values
    c_1d_aligned = align_htf_to_ltf(prices, df_1d, c_1d)
    
    # Pre-compute 1d EMA(200) for trend filter
    ema200_1d = pd.Series(c_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    for i in range(200, n):  # Start after EMA200 warmup
        # Skip if any required data is invalid
        if (np.isnan(willr[i]) or np.isnan(volume_20_avg[i]) or 
            np.isnan(ema200_1d_aligned[i]) or np.isnan(c_1d_aligned[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new mean reversion entries
            # Long when oversold AND in 1d uptrend with volume spike
            if (willr[i] < -80 and 
                prices['close'].iloc[i] > ema200_1d_aligned[i] and 
                vol_spike.iloc[i]):
                position = 1
                signals[i] = 0.25
            # Short when overbought AND in 1d downtrend with volume spike
            elif (willr[i] > -20 and 
                  prices['close'].iloc[i] < ema200_1d_aligned[i] and 
                  vol_spike.iloc[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit when Williams %R crosses -50 (mean reversion complete)
            exit_signal = False
            if position == 1:  # Long position
                if willr[i] > -50:
                    exit_signal = True
            elif position == -1:  # Short position
                if willr[i] < -50:
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