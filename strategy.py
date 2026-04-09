#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 1d Williams %R mean reversion
# - Elder Ray (Bull/Bear power) from 6h: measures buying/selling pressure
# - Williams %R from 1d: identifies overbought/oversold conditions for mean reversion
# - Long when Bull Power > 0 and Williams %R < -80 (oversold with buying pressure)
# - Short when Bear Power < 0 and Williams %R > -20 (overbought with selling pressure)
# - Exit when Elder Ray power reverses or Williams %R returns to neutral (-50)
# - Position size: 0.25 to manage drawdown in volatile 6h timeframe
# - Target: 50-100 total trades over 4 years (12-25/year) to minimize fee drag
# - Works in bull markets (buy dips in uptrend) and bear markets (sell rallies in downtrend)
# - Combines momentum (Elder Ray) with mean reversion (Williams %R) for robust signals

name = "6h_1d_elder_ray_williamsr_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Pre-compute 1d indicators
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d Williams %R(14)
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close_1d) / (highest_high_14 - lowest_low_14 + 1e-10)
    
    # Align 1d Williams %R to 6h
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # 6h price data for Elder Ray
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # 6h EMA(13) for Elder Ray (standard period)
    close_series = pd.Series(close)
    ema_13 = close_series.ewm(span=13, min_periods=13, adjust=False).mean().values
    
    # 6h Elder Ray components
    bull_power = high - ema_13     # Buying pressure: high minus EMA
    bear_power = low - ema_13      # Selling pressure: low minus EMA
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(13, n):  # Start after EMA warmup
        # Skip if any required data is invalid
        if (np.isnan(williams_r_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(ema_13[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit when Bull Power turns negative OR Williams %R returns to neutral
            if bull_power[i] <= 0 or williams_r_aligned[i] >= -50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit when Bear Power turns positive OR Williams %R returns to neutral
            if bear_power[i] >= 0 or williams_r_aligned[i] <= -50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for mean reversion Elder Ray signals
            # Long: Bull Power positive AND Williams %R oversold (< -80)
            if bull_power[i] > 0 and williams_r_aligned[i] < -80:
                position = 1
                signals[i] = 0.25
            # Short: Bear Power negative AND Williams %R overbought (> -20)
            elif bear_power[i] < 0 and williams_r_aligned[i] > -20:
                position = -1
                signals[i] = -0.25
    
    return signals