#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams %R with 1-week EMA34 trend filter and volume confirmation (>1.5x 20 EMA volume)
# Uses 1d Williams %R(14) for mean reversion signals - effective in ranging markets
# 1-week EMA34 ensures alignment with higher timeframe trend to avoid counter-trend whipsaws
# Volume confirmation filters false signals (>1.5x average volume) - balanced to reduce overtrading
# Discrete sizing 0.25 minimizes fee churn while maintaining profitability
# Target: 30-100 total trades over 4 years = 7-25/year for 1d timeframe
# Works in bull markets (mean reversion from oversold in uptrend) and bear markets (mean reversion from overbought in downtrend)
# Focus on BTC/ETH by requiring 1w trend alignment (avoids SOL-only bias)

name = "1d_WilliamsR_1wEMA34_VolumeConfirm"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:  # Need enough data for EMA34 calculation
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA(34) trend filter from prior completed 1w bar
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_shifted = np.roll(ema_34_1w, 1)
    ema_34_1w_shifted[0] = np.nan
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w_shifted)
    
    # Volume confirmation: 20-period EMA of volume
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate 1d Williams %R(14) from prior completed 1d bar
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    
    # Shift by 1 to use only prior completed 1d bar (no look-ahead)
    williams_r_shifted = np.roll(williams_r, 1)
    williams_r_shifted[0] = np.nan
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_ema_20[i]) or 
            np.isnan(williams_r_shifted[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Williams %R < -80 (oversold) AND price > 1w EMA34 AND volume spike
            if williams_r_shifted[i] < -80 and close[i] > ema_34_1w_aligned[i] and volume[i] > (1.5 * vol_ema_20[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Williams %R > -20 (overbought) AND price < 1w EMA34 AND volume spike
            elif williams_r_shifted[i] > -20 and close[i] < ema_34_1w_aligned[i] and volume[i] > (1.5 * vol_ema_20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R > -50 (exiting oversold) OR price crosses below 1w EMA34
            if williams_r_shifted[i] > -50 or close[i] < ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R < -50 (exiting overbought) OR price crosses above 1w EMA34
            if williams_r_shifted[i] < -50 or close[i] > ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals