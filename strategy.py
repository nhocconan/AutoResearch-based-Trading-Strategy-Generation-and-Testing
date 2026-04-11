#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h timeframe with 4h and 1d multi-timeframe confirmation.
# Uses 4h RSI for trend direction and 1d volume filter for institutional participation.
# Takes long positions when 4h RSI > 60 (bullish) and price pulls back to 1h EMA(20) with volume confirmation.
# Takes short positions when 4h RSI < 40 (bearish) and price rallies to 1h EMA(20) with volume confirmation.
# Designed for 15-30 trades/year (60-120 total over 4 years) with 0.20 position sizing.
# The 4h RSI filter reduces whipsaw in sideways markets, while volume confirmation ensures quality setups.

name = "1h_4h1d_rsi_volume_pullback_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 4h and 1d data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 20 or len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 4h RSI(14) for trend direction
    close_4h = df_4h['close'].values
    delta = np.diff(close_4h, prepend=close_4h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi_4h = 100 - (100 / (1 + rs))
    rsi_4h[:13] = np.nan  # Not enough data for first 13 periods
    
    # Calculate 1h EMA(20) for pullback entries
    close_series = pd.Series(close)
    ema_20 = close_series.ewm(span=20, adjust=False).mean().values
    
    # Calculate 1d average volume (20-period) for institutional confirmation
    volume_1d = df_1d['volume'].values
    vol_avg_20 = np.full_like(volume_1d, np.nan, dtype=float)
    for i in range(19, len(volume_1d)):
        vol_avg_20[i] = np.mean(volume_1d[i-19:i+1])
    
    # Align all indicators to 1h timeframe
    rsi_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_4h)
    ema_20_aligned = align_htf_to_ltf(prices, df_4h, ema_20)  # Using 4h index for EMA since it's calculated on close
    vol_avg_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(rsi_4h_aligned[i]) or np.isnan(ema_20_aligned[i]) or 
            np.isnan(vol_avg_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        # Volume filter: current volume > 1.3 * daily average volume
        vol_filter = volume[i] > 1.3 * vol_avg_aligned[i]
        
        # Determine 4h RSI trend conditions
        rsi_bullish = rsi_4h_aligned[i] > 60
        rsi_bearish = rsi_4h_aligned[i] < 40
        
        # Long setup: bullish 4h trend + price pullback to EMA(20) with volume
        long_setup = (rsi_bullish and 
                     low[i] <= ema_20_aligned[i] and 
                     vol_filter)
        
        # Short setup: bearish 4h trend + price rally to EMA(20) with volume
        short_setup = (rsi_bearish and 
                      high[i] >= ema_20_aligned[i] and 
                      vol_filter)
        
        # Exit conditions: reverse signal or loss of momentum
        exit_long = (position == 1 and 
                    (rsi_4h_aligned[i] < 50 or  # RSI turns bearish
                     high[i] >= ema_20_aligned[i] * 1.02))  # 2% above EMA
        
        exit_short = (position == -1 and 
                     (rsi_4h_aligned[i] > 50 or  # RSI turns bullish
                      low[i] <= ema_20_aligned[i] * 0.98))  # 2% below EMA
        
        # Enter long on setup
        if long_setup and position != 1:
            position = 1
            signals[i] = 0.20
        # Enter short on setup
        elif short_setup and position != -1:
            position = -1
            signals[i] = -0.20
        # Exit long
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        # Exit short
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.20 if position == 1 else (-0.20 if position == -1 else 0.0)
    
    return signals