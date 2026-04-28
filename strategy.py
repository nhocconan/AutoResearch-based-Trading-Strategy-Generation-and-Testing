#!/usr/bin/env python3
"""
6h_RSI_Extreme_4hTrend_Filter_VolumeSpike
Hypothesis: On 6h timeframe, RSI(14) extremes (overbought/oversold) combined with 4h EMA20 trend filter and volume spike confirmation will capture mean-reversion opportunities in both bull and bear markets. The 4h EMA20 acts as a trend filter to avoid counter-trend trades, while volume spikes confirm institutional participation at extreme RSI levels. Targets 15-25 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4h EMA20 for trend filter
    close_4h = df_4h['close'].values
    ema_20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # Calculate RSI(14) on 6h data
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate 20-period volume MA for volume spike confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(ema_20_4h_aligned[i]) or np.isnan(rsi[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        # Trend direction from 4h EMA20
        trend_up = close[i] > ema_20_4h_aligned[i]
        trend_down = close[i] < ema_20_4h_aligned[i]
        
        # Volume confirmation: >2.0x 20-period MA
        vol_confirm = volume[i] > (2.0 * vol_ma_20[i])
        
        # RSI extreme conditions
        rsi_overbought = rsi[i] >= 70
        rsi_oversold = rsi[i] <= 30
        
        # Entry logic: mean reversion at extremes with trend filter and volume confirmation
        long_entry = rsi_oversold and trend_up and vol_confirm
        short_entry = rsi_overbought and trend_down and vol_confirm
        
        # Exit logic: RSI returns to neutral zone or trend reversal
        long_exit = (rsi[i] >= 50) or (not trend_up)
        short_exit = (rsi[i] <= 50) or (not trend_down)
        
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

name = "6h_RSI_Extreme_4hTrend_Filter_VolumeSpike"
timeframe = "6h"
leverage = 1.0