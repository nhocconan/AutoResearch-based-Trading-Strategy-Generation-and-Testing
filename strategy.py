#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R (14) mean reversion with 1d EMA34 trend filter and volume confirmation (>2.0x 20 EMA volume)
# Williams %R identifies overbought/oversold conditions - long when %R < -80 (oversold) and price > 1d EMA34
# Short when %R > -20 (overbought) and price < 1d EMA34
# 1d EMA34 ensures alignment with higher timeframe trend to avoid counter-trend whipsaws
# Volume confirmation filters false signals (>2.0x average volume) - tight threshold to hit 12-37 trades/year target
# Discrete sizing 0.25 minimizes fee churn while maintaining profitability
# Works in bull markets (buy dips in uptrend) and bear markets (sell rallies in downtrend)
# Focus on BTC/ETH by requiring 1d trend alignment (avoids SOL-only bias)

name = "12h_WilliamsR_1dEMA34_VolumeConfirm_v1"
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
    
    # Get 1d data for EMA calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # Need enough data for EMA34 calculation
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA(34) trend filter from prior completed 1d bar
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_shifted = np.roll(ema_34_1d, 1)
    ema_34_1d_shifted[0] = np.nan
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d_shifted)
    
    # Volume confirmation: 20-period EMA of volume
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate Williams %R(14) - using prior completed 12h bar
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Shift to use only completed bars (avoid look-ahead)
    highest_high_shifted = np.roll(highest_high, 1)
    lowest_low_shifted = np.roll(lowest_low, 1)
    highest_high_shifted[0] = np.nan
    lowest_low_shifted[0] = np.nan
    
    # Williams %R calculation
    williams_r = np.where(
        (highest_high_shifted - lowest_low_shifted) != 0,
        ((highest_high_shifted - close) / (highest_high_shifted - lowest_low_shifted)) * -100,
        np.nan
    )
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ema_20[i]) or 
            np.isnan(williams_r[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Williams %R < -80 (oversold) AND price > 1d EMA34 AND volume spike
            if williams_r[i] < -80.0 and close[i] > ema_34_1d_aligned[i] and volume[i] > (2.0 * vol_ema_20[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Williams %R > -20 (overbought) AND price < 1d EMA34 AND volume spike
            elif williams_r[i] > -20.0 and close[i] < ema_34_1d_aligned[i] and volume[i] > (2.0 * vol_ema_20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R > -50 (return from oversold) OR price crosses below 1d EMA34
            if williams_r[i] > -50.0 or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R < -50 (return from overbought) OR price crosses above 1d EMA34
            if williams_r[i] < -50.0 or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals