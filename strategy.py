#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using 1-week Exponential Moving Average (EMA) as trend filter
# and 1-week Average True Range (ATR) for volatility-adjusted breakout detection.
# Long when price breaks above weekly EMA by >1.5 * weekly ATR with volume confirmation (>1.2x average volume).
# Short when price breaks below weekly EMA by >1.5 * weekly ATR with volume confirmation.
# Exit when price crosses back over weekly EMA.
# Weekly EMA provides robust trend direction, ATR normalization adapts to volatility,
# volume confirmation reduces false breakouts. Weekly timeframe reduces noise and
# aligns with institutional flows, effective in both bull and bear markets.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE for EMA, ATR, and average volume
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:  # Need enough for EMA(20) and ATR(14)
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate EMA (20) on weekly close
    ema_20 = np.full_like(close_1w, np.nan)
    ema_20[19] = np.mean(close_1w[:20])  # Simple average for first value
    alpha = 2 / (20 + 1)
    for i in range(20, len(close_1w)):
        ema_20[i] = alpha * close_1w[i] + (1 - alpha) * ema_20[i-1]
    
    # Calculate ATR (14) on weekly data
    # True Range
    tr1 = high_1w[1:] - low_1w[1:]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # ATR = smoothed TR (Wilder's smoothing)
    atr = np.full_like(tr, np.nan)
    atr[13] = np.nanmean(tr[1:14])  # First ATR: simple average of first 14 TR
    for i in range(14, len(tr)):
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # Calculate average volume (20-period) for volume confirmation
    avg_volume = np.full_like(volume_1w, np.nan)
    for i in range(19, len(volume_1w)):
        avg_volume[i] = np.mean(volume_1w[i-19:i+1])
    
    # Align indicators to 1d timeframe
    ema_20_aligned = align_htf_to_ltf(prices, df_1w, ema_20)
    atr_aligned = align_htf_to_ltf(prices, df_1w, atr)
    avg_volume_aligned = align_htf_to_ltf(prices, df_1w, avg_volume)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(30, 20)  # Need EMA and ATR periods
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_20_aligned[i]) or 
            np.isnan(atr_aligned[i]) or
            np.isnan(avg_volume_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.2 * average weekly volume
        volume_confirmed = volume[i] > (1.2 * avg_volume_aligned[i])
        
        if position == 0:
            # Look for breakout entries
            # Long: price breaks above EMA by >1.5*ATR AND volume confirmed
            if (close[i] > ema_20_aligned[i] + (1.5 * atr_aligned[i]) and 
                volume_confirmed):
                position = 1
                signals[i] = position_size
            # Short: price breaks below EMA by >1.5*ATR AND volume confirmed
            elif (close[i] < ema_20_aligned[i] - (1.5 * atr_aligned[i]) and 
                  volume_confirmed):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses back below EMA
            if close[i] < ema_20_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses back above EMA
            if close[i] > ema_20_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_1w_EMA_ATR_Volume_Breakout_v1"
timeframe = "1d"
leverage = 1.0