#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams %R reversal with 1w EMA50 trend filter and volume confirmation
# Uses 1d primary timeframe targeting 7-25 trades/year (30-100 total over 4 years)
# 1w EMA50 ensures alignment with weekly trend to avoid counter-trend entries
# Williams %R(14) identifies overbought/oversold conditions for mean reversion
# Long: %R < -80 (oversold) + price > 1w EMA50 + volume confirmation
# Short: %R > -20 (overbought) + price < 1w EMA50 + volume confirmation
# Discrete position sizing (0.25) minimizes fee churn while maintaining adequate exposure
# Works in bull (continuation on dips) and bear (mean reversion on rallies) markets

name = "1d_WilliamsR_1wEMA50_Trend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Williams %R(14) on 1d data
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when high == low
    williams_r[highest_high == lowest_low] = -50
    
    # 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1w EMA50 to 1d timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: volume > 1.5 * 20-period EMA (1d)
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup: need sufficient data for all indicators
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(williams_r[i]) or np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: oversold (%R < -80) + price above 1w EMA50 + volume spike
            if williams_r[i] < -80 and close[i] > ema_50_1w_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: overbought (%R > -20) + price below 1w EMA50 + volume spike
            elif williams_r[i] > -20 and close[i] < ema_50_1w_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Williams %R returns above -50 (mean reversion) or price below 1w EMA50
            if williams_r[i] > -50 or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Williams %R returns below -50 (mean reversion) or price above 1w EMA50
            if williams_r[i] < -50 or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals