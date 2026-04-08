#!/usr/bin/env python3
"""
4h Bollinger Band Width Squeeze Breakout with 12h Volume Confirmation
Hypothesis: Bollinger Band Width squeeze indicates low volatility and impending breakout.
Breakout occurs when price closes outside bands with high volume (>1.5x 20-period average).
Filtered by 12h EMA trend to avoid counter-trend trades. Works in bull/bear by aligning with higher timeframe trend.
Targets 20-50 trades/year on 4h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_bollinger_squeeze_breakout_12h_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h EMA(50) for trend filter
    df_12h = get_htf_data(prices, '12h')
    ema_50_12h = df_12h['close'].ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Bollinger Bands (20, 2)
    bb_period = 20
    bb_std = 2
    sma = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    std = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper = sma + (bb_std * std)
    lower = sma - (bb_std * std)
    bb_width = upper - lower
    
    # Bollinger Band Width Squeeze: BBW < 50th percentile of last 50 periods
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=50, min_periods=20).quantile(0.5).values
    squeeze = bb_width < bb_width_percentile
    
    # Volume filter (>1.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after BB width percentile warmup
        # Skip if any required data is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(sma[i]) or 
            np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(squeeze[i]) or np.isnan(vol_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below middle band (mean reversion) OR squeeze ends
            if (close[i] <= sma[i] or not squeeze[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above middle band OR squeeze ends
            if (close[i] >= sma[i] or not squeeze[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: squeeze active, price breaks above upper band, uptrend, volume
            if (squeeze[i] and 
                close[i] > upper[i] and 
                close[i] > ema_50_12h_aligned[i] and 
                vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short: squeeze active, price breaks below lower band, downtrend, volume
            elif (squeeze[i] and 
                  close[i] < lower[i] and 
                  close[i] < ema_50_12h_aligned[i] and 
                  vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals