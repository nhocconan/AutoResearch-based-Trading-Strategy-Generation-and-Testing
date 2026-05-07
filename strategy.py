#!/usr/bin/env python3
# 4H_RSI_Squeeze_Pattern
# Hypothesis: Combines RSI momentum squeeze with Bollinger Band compression and volume confirmation.
# Enters long when RSI shows bullish divergence during low volatility squeeze, short on bearish divergence.
# Uses 4h timeframe with daily trend filter to avoid counter-trend trades.
# Designed for 20-40 trades/year to minimize fee drag while capturing mean reversion in ranging markets.

name = "4H_RSI_Squeeze_Pattern"
timeframe = "4h"
leverage = 1.0

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
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 20-period RSI
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate Bollinger Bands (20, 2)
    ma20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper = ma20 + 2 * std20
    lower = ma20 - 2 * std20
    bb_width = (upper - lower) / ma20  # Normalized bandwidth
    
    # Daily EMA50 trend filter
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50)  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(rsi[i]) or np.isnan(bb_width[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma20[i]) or
            vol_ma20[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Squeeze condition: low volatility (BB width in lowest 20% of last 50 periods)
        bb_width_percentile = pd.Series(bb_width[max(0, i-49):i+1]).rank(pct=True).iloc[-1] if i >= 49 else 0.5
        squeeze_condition = bb_width_percentile < 0.2
        
        # Volume confirmation
        volume_filter = volume[i] > 1.5 * vol_ma20[i]
        
        if position == 0:
            # RSI conditions for entry
            rsi_rising = rsi[i] > rsi[i-1]
            rsi_falling = rsi[i] < rsi[i-1]
            rsi_oversold = rsi[i] < 30
            rsi_overbought = rsi[i] > 70
            
            # Long: RSI rising from oversold during squeeze + uptrend + volume
            if (rsi_rising and rsi_oversold and squeeze_condition and 
                close[i] > ema50_1d_aligned[i] and volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: RSI falling from overbought during squeeze + downtrend + volume
            elif (rsi_falling and rsi_overbought and squeeze_condition and 
                  close[i] < ema50_1d_aligned[i] and volume_filter):
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit conditions: RSI mean reversion or trend change
            rsi_overbought_exit = rsi[i] > 70
            rsi_oversold_exit = rsi[i] < 30
            trend_change = (position == 1 and close[i] < ema50_1d_aligned[i]) or \
                          (position == -1 and close[i] > ema50_1d_aligned[i])
            
            if (position == 1 and rsi_overbought_exit) or \
               (position == -1 and rsi_oversold_exit) or \
               trend_change:
                signals[i] = 0.0
                position = 0
            else:
                # Maintain position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals