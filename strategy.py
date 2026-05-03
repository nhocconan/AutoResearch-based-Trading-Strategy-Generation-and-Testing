#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R reversal with 1d EMA50 trend filter and volume confirmation.
# Long when Williams %R crosses above -80 from below in bull trend (close > 1d EMA50) with volume > 1.5x 24-period MA.
# Short when Williams %R crosses below -20 from above in bear trend (close < 1d EMA50) with volume spike.
# Uses discrete position sizing (0.25) to minimize fee churn. 1d EMA50 provides strong trend filter.
# Williams %R identifies overbought/oversold levels for mean reversion within the trend.
# Volume confirmation ensures institutional participation. Target: 50-150 total trades over 4 years (12-37/year).
# Works in both bull and bear markets: trend filter ensures we only trade in direction of 1d momentum,
# while Williams %R provides precise reversal entries within that trend.

name = "12h_WilliamsR_1dEMA50_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter and Williams %R
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Get 1d data for Williams %R (14-period)
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate Williams %R for each 1d bar
    highest_high_14 = pd.Series(df_1d['high'].values).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(df_1d['low'].values).rolling(window=14, min_periods=14).min().values
    close_1d = df_1d['close'].values
    
    # Williams %R = -100 * (HH - Close) / (HH - LL)
    williams_r = -100 * (highest_high_14 - close_1d) / (highest_high_14 - lowest_low_14)
    # Handle division by zero (when HH == LL)
    williams_r = np.where((highest_high_14 - lowest_low_14) == 0, -50, williams_r)
    
    # Align Williams %R to 12h timeframe (no additional delay needed as it's based on completed 1d bar)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Volume regime: current 12h volume > 1.5x 24-period MA (more conservative than 2.0x)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (1.5 * vol_ma_24)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(williams_r_aligned[i]) or 
            np.isnan(vol_ma_24[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        ema_trend = ema_50_1d_aligned[i]
        williams_r_val = williams_r_aligned[i]
        vol_spike = volume_spike[i]
        
        # Determine trend regime
        is_bull_trend = close_val > ema_trend
        is_bear_trend = close_val < ema_trend
        
        # Williams %R reversal conditions
        williams_r_oversold = williams_r_val < -80  # Oversold
        williams_r_overbought = williams_r_val > -20  # Overbought
        
        # Entry logic
        if position == 0:
            if is_bull_trend and williams_r_oversold and vol_spike:
                signals[i] = 0.25
                position = 1
            elif is_bear_trend and williams_r_overbought and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Williams %R becomes overbought OR trend reversal
            if williams_r_val > -20 or close_val < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R becomes oversold OR trend reversal
            if williams_r_val < -80 or close_val > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals