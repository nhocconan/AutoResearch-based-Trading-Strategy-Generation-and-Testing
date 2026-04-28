#!/usr/bin/env python3
"""
6h_RSI_Extreme_12hTrend_VolumeSpike
Hypothesis: Enter long when RSI(14) < 25 (deep oversold) on 6h with 12h uptrend (price > EMA50) and volume spike (>2x 20-period MA). Enter short when RSI > 75 (overbought) with 12h downtrend (price < EMA50) and volume spike. This targets mean-reversion bounces within established trends, filtering counter-trend noise. Works in both bull/bear markets by aligning with intermediate trend while capturing exhaustion moves.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # RSI(14) on 6h
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(100).values  # Handle div/0
    
    # Volume confirmation: >2.0x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(rsi[i]) or np.isnan(ema_50_12h_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend from 12h: price vs EMA50
        trend_up = close[i] > ema_50_12h_aligned[i]
        trend_down = close[i] < ema_50_12h_aligned[i]
        
        # Volume confirmation
        vol_confirm = volume[i] > (2.0 * vol_ma_20[i])
        
        # RSI extremes
        rsi_oversold = rsi[i] < 25
        rsi_overbought = rsi[i] > 75
        
        # Entry logic: RSI extreme in direction of 12h trend with volume
        long_entry = vol_confirm and trend_up and rsi_oversold
        short_entry = vol_confirm and trend_down and rsi_overbought
        
        # Exit logic: RSI returns to neutral zone (40-60) or trend change
        long_exit = (rsi[i] > 40) or (not trend_up)
        short_exit = (rsi[i] < 60) or (not trend_down)
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_RSI_Extreme_12hTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0