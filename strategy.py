#!/usr/bin/env python3
"""
1D_Camarilla_R3S3_Breakout_1wEMA10_Trend
Hypothesis: Weekly EMA10 defines long-term trend, daily Camarilla R3/S3 levels act as strong support/resistance.
In bull markets, buy breakouts above R3 with weekly uptrend. In bear markets, sell breakdowns below S3 with weekly downtrend.
Volume spike confirms institutional interest. Uses 1d timeframe for low turnover and minimal fee drag.
Target: 15-25 trades/year, low turnover to minimize fee drag in ranging 2025 market.
"""

name = "1D_Camarilla_R3S3_Breakout_1wEMA10_Trend"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema10_1w = pd.Series(close_1w).ewm(span=10, min_periods=10, adjust=False).mean().values
    ema10_1w_aligned = align_htf_to_ltf(prices, df_1w, ema10_1w)
    
    # Load daily data ONCE for Camarilla levels
    high_1d = high
    low_1d = low
    close_1d = close
    
    # Calculate Camarilla levels: R3, S3 (outer levels for fewer, stronger signals)
    hl_range = high_1d - low_1d
    r3 = close_1d + hl_range * 1.5000
    s3 = close_1d - hl_range * 1.5000
    
    # Volume filter: 20-period EMA for spike detection (using daily volume)
    vol_ema20 = pd.Series(volume).ewm(span=20, min_periods=20, adjust=False).mean().values
    volume_ok = volume > vol_ema20 * 1.5
    
    # Fixed position size to minimize churn
    position_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(ema10_1w_aligned[i]) or np.isnan(r3[i]) or 
            np.isnan(s3[i]) or np.isnan(volume_ok[i])):
            if position == 1:
                signals[i] = 0.0
            elif position == -1:
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        # Conditions
        price_above_ema1w = close[i] > ema10_1w_aligned[i]
        price_below_ema1w = close[i] < ema10_1w_aligned[i]
        breakout_long = close[i] > r3[i]
        breakout_short = close[i] < s3[i]
        
        if position == 0:
            # Long: Price breaks above R3 + above weekly EMA10 + volume spike
            if breakout_long and price_above_ema1w and volume_ok[i]:
                signals[i] = position_size
                position = 1
            # Short: Price breaks below S3 + below weekly EMA10 + volume spike
            elif breakout_short and price_below_ema1w and volume_ok[i]:
                signals[i] = -position_size
                position = -1
        else:
            # Exit conditions - simplified to reduce churn
            if position == 1:
                # Exit: Price crosses below S3 OR trend reverses (close below weekly EMA)
                if close[i] < s3[i] or close[i] < ema10_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = position_size
            elif position == -1:
                # Exit: Price crosses above R3 OR trend reverses (close above weekly EMA)
                if close[i] > r3[i] or close[i] > ema10_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -position_size
    
    return signals