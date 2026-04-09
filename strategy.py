#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R mean reversal with 1d trend filter and volume confirmation
# - Primary signal: 4h Williams %R < -80 (oversold) for long, > -20 (overbought) for short
# - Trend filter: 1d close > 1d EMA50 for long bias, < 1d EMA50 for short bias
# - Volume confirmation: 4h volume > 4h volume EMA20 (avoid low-participation signals)
# - Position size: 0.25 discrete level to minimize fee churn
# - Target: 20-50 trades/year (80-200 total over 4 years) per 4h strategy guidelines
# - Works in bull/bear: Williams %R captures exhaustion, 1d EMA50 filters counter-trend trades

name = "4h_1d_williamsr_ema_volume_v1"
timeframe = "4h"
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
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 4h timeframe (completed 1d bar only)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 4h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 4h Williams %R calculation
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high_14 - close) / (highest_high_14 - lowest_low_14) * -100
    williams_r = np.where(highest_high_14 == lowest_low_14, -50, williams_r)  # neutral when range=0
    
    # 4h volume confirmation: volume > volume EMA20
    volume_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_confirm = volume > volume_ema_20
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_aligned[i]) or
            np.isnan(williams_r[i]) or
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Williams %R > -50 (exit oversold) OR price closes below 1d EMA50
            if williams_r[i] > -50 or close[i] < ema_50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Williams %R < -50 (exit overbought) OR price closes above 1d EMA50
            if williams_r[i] < -50 or close[i] > ema_50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Williams %R extremes with volume confirmation and trend filter
            # Long: Williams %R < -80 (oversold) AND volume confirmation AND price > 1d EMA50 (uptrend)
            if williams_r[i] < -80 and volume_confirm[i] and close[i] > ema_50_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Short: Williams %R > -20 (overbought) AND volume confirmation AND price < 1d EMA50 (downtrend)
            elif williams_r[i] > -20 and volume_confirm[i] and close[i] < ema_50_aligned[i]:
                position = -1
                signals[i] = -0.25
    
    return signals