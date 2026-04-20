#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams %R mean reversion with weekly trend filter and volume confirmation.
# Williams %R identifies overbought/oversold conditions; weekly EMA200 provides trend direction
# to avoid counter-trend trades. Volume spike confirms institutional participation.
# Works in bull markets (buy dips in uptrend) and bear markets (sell rallies in downtrend).
# Target: 15-25 trades per year to minimize fee drag.

name = "1d_WilliamsR_WeeklyEMA200_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data ONCE before loop for EMA200 trend
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 2:
        return np.zeros(n)
    
    # === Weekly EMA200 for trend direction ===
    close_weekly = df_weekly['close'].values
    ema_200 = pd.Series(close_weekly).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_aligned = align_htf_to_ltf(prices, df_weekly, ema_200)
    
    # === Daily Williams %R (14-period) ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate highest high and lowest low over 14 days
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    williams_r = ((highest_high - close) / (highest_high - lowest_low)) * -100
    
    # === Daily Volume confirmation ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Get values
        wr = williams_r[i]
        ema_val = ema_200_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(wr) or np.isnan(ema_val) or np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Williams %R oversold (< -80) + uptrend (price > EMA200) + volume spike
            if wr < -80 and close[i] > ema_val and vol_ratio_val > 1.5:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (> -20) + downtrend (price < EMA200) + volume spike
            elif wr > -20 and close[i] < ema_val and vol_ratio_val > 1.5:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Williams %R returns to neutral (> -50) or trend reversal
            if wr > -50 or close[i] < ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Williams %R returns to neutral (< -50) or trend reversal
            if wr < -50 or close[i] > ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals