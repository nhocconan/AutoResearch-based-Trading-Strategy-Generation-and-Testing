#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with 1d EMA trend filter
# - Long: Williams %R(14) crosses above -80 (oversold) + price > 1d EMA(50) (uptrend)
# - Short: Williams %R(14) crosses below -20 (overbought) + price < 1d EMA(50) (downtrend)
# - Uses discrete position sizing (0.25) to minimize fee churn
# - No stoploss - exits on opposite signal to reduce whipsaw
# - Designed for 6h timeframe: targets 12-37 trades/year to avoid fee drag
# - Works in bull/bear markets: trend filter prevents counter-trend trades, Williams %R captures mean reversion in ranging markets

name = "6h_1d_williamsr_mean_reversion_v3"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Pre-compute 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Pre-compute 6h Williams %R(14)
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    
    # Calculate highest high and lowest low over 14 periods
    highest_high = pd.Series(high_6h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_6h).rolling(window=14, min_periods=14).min().values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    williams_r = np.where((highest_high - lowest_low) != 0, 
                          ((highest_high - close_6h) / (highest_high - lowest_low)) * -100, 
                          -50)  # neutral when range is zero
    
    # Williams %R cross signals
    williams_r_prev = np.roll(williams_r, 1)
    williams_r_prev[0] = np.nan
    
    # Long signal: Williams %R crosses above -80 (from below)
    long_signal = (williams_r > -80) & (williams_r_prev <= -80)
    # Short signal: Williams %R crosses below -20 (from above)
    short_signal = (williams_r < -20) & (williams_r_prev >= -20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(long_signal[i]) or np.isnan(short_signal[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: short signal triggers
            if short_signal[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: long signal triggers
            if long_signal[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Williams %R signals with trend filter
            if long_signal[i] and close_6h[i] > ema_50_aligned[i]:
                position = 1
                signals[i] = 0.25
            elif short_signal[i] and close_6h[i] < ema_50_aligned[i]:
                position = -1
                signals[i] = -0.25
    
    return signals