#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R Extreme Reversal with 1w EMA Trend Filter and Volume Confirmation
# Long when: Williams %R(14) < -80 (oversold) AND price > 1w EMA50 (uptrend) AND 6h volume > 1.5x 20-period average
# Short when: Williams %R(14) > -20 (overbought) AND price < 1w EMA50 (downtrend) AND 6h volume > 1.5x 20-period average
# Exit when Williams %R crosses back through -50 (mean reversion) or opposite extreme
# Williams %R identifies exhaustion points in both bull and bear markets
# 1w EMA50 filters for primary trend alignment to avoid counter-trend trades
# Volume confirmation ensures institutional participation at reversal points
# Target: 80-120 total trades over 4 years (20-30/year) with discrete sizing 0.25

name = "6h_WilliamsR_Extreme_Reversal_1wEMA50_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data ONCE before loop for EMA50
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:  # Need enough for EMA50
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA50
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Williams %R(14) on 6h data
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate 6h volume spike (current volume > 1.5x 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Check conditions
        vol_cond = bool(vol_spike[i])
        above_ema = close[i] > ema_50_1w_aligned[i]
        below_ema = close[i] < ema_50_1w_aligned[i]
        
        if position == 0:
            # Long: Oversold + Uptrend + Volume Spike
            if williams_r[i] < -80 and above_ema and vol_cond:
                signals[i] = 0.25
                position = 1
            # Short: Overbought + Downtrend + Volume Spike
            elif williams_r[i] > -20 and below_ema and vol_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R crosses above -50 (mean reversion) or opposite extreme
            if williams_r[i] > -50 or williams_r[i] > -20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R crosses below -50 (mean reversion) or opposite extreme
            if williams_r[i] < -50 or williams_r[i] < -80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals