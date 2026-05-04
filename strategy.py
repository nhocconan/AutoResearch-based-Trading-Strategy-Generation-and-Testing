#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R mean reversion with 1w EMA50 trend filter and volume confirmation
# Williams %R(14) identifies overbought/oversold conditions: < -80 = oversold, > -20 = overbought
# Long when %R crosses above -80 (oversold recovery) AND price > 1w EMA50 (uptrend) AND volume > 1.5x average
# Short when %R crosses below -20 (overbought correction) AND price < 1w EMA50 (downtrend) AND volume > 1.5x average
# Uses 1w EMA50 for higher timeframe trend alignment to avoid counter-trend whipsaws
# Volume confirmation filters false signals (>1.5x 20-period volume EMA)
# Discrete sizing 0.25 minimizes fee churn while maintaining profitability
# Target: 50-150 total trades over 4 years = 12-37/year for 12h timeframe
# Works in bull markets (buying dips in uptrend) and bear markets (selling rallies in downtrend)
# Focus on BTC/ETH by requiring 1w trend alignment (avoids SOL-only bias)

name = "12h_WilliamsR_1wEMA50_VolumeConfirm"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:  # Need enough data for EMA50 calculation
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA(50) trend filter from prior completed 1w bar
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_shifted = np.roll(ema_50_1w, 1)
    ema_50_1w_shifted[0] = np.nan
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w_shifted)
    
    # Williams %R(14) calculation on 12h data
    # %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)  # avoid division by zero
    
    # Volume confirmation: 20-period EMA of volume
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Williams %R crosses above -80 (oversold recovery) 
            # AND price > 1w EMA50 (uptrend) AND volume spike
            if (williams_r[i] > -80 and williams_r[i-1] <= -80 and 
                close[i] > ema_50_1w_aligned[i] and 
                volume[i] > (1.5 * vol_ema_20[i])):
                signals[i] = 0.25
                position = 1
            # Short conditions: Williams %R crosses below -20 (overbought correction)
            # AND price < 1w EMA50 (downtrend) AND volume spike
            elif (williams_r[i] < -20 and williams_r[i-1] >= -20 and 
                  close[i] < ema_50_1w_aligned[i] and 
                  volume[i] > (1.5 * vol_ema_20[i])):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R crosses below -50 (momentum loss) 
            # OR price crosses below 1w EMA50 (trend change)
            if williams_r[i] < -50 or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R crosses above -50 (momentum loss)
            # OR price crosses above 1w EMA50 (trend change)
            if williams_r[i] > -50 or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals