#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d ATR breakout with 1w trend filter and volume confirmation.
# Uses daily ATR-based breakout channels (ATR multiplier = 2.0) to capture momentum.
# Weekly EMA(50) as trend filter ensures alignment with higher timeframe direction.
# Volume confirmation filters false breakouts (volume > 1.5x 20-day average).
# Designed for low trade frequency (target: 10-25 trades/year) to minimize fee drag.
# Works in bull markets via breakout continuation and in bear via short breakdowns.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1-week data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # EMA(50) for 1w trend filter
    ema50_1w = np.zeros(len(close_1w))
    ema_multiplier = 2 / (50 + 1)
    ema50_1w[0] = close_1w[0]
    for i in range(1, len(close_1w)):
        ema50_1w[i] = (close_1w[i] - ema50_1w[i-1]) * ema_multiplier + ema50_1w[i-1]
    
    # Align 1w EMA to 1d timeframe
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # True Range and ATR(14) for breakout channels
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]  # First bar: no previous close
    atr = np.zeros(n)
    atr[13] = np.mean(tr[:14])  # Seed ATR with first 14 values
    for i in range(14, n):
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14  # Wilder's smoothing
    
    # ATR-based breakout channels
    upper_channel = np.roll(close, 1) + 2.0 * atr  # Previous close + 2*ATR
    lower_channel = np.roll(close, 1) - 2.0 * atr  # Previous close - 2*ATR
    
    # Average volume (20-day) for volume confirmation
    avg_volume = np.zeros(n)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):
        # Skip if any required data is not ready
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        prev_close = close[i-1]
        vol = volume[i]
        avg_vol = avg_volume[i]
        ema_trend = ema50_1w_aligned[i]
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirm = vol > 1.5 * avg_vol
        
        if position == 0:
            # Long: Break above upper channel + above 1w EMA50 + volume confirmation
            if (price > upper_channel[i] and
                price > ema_trend and
                volume_confirm):
                position = 1
                signals[i] = position_size
            # Short: Break below lower channel + below 1w EMA50 + volume confirmation
            elif (price < lower_channel[i] and
                  price < ema_trend and
                  volume_confirm):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Price breaks below lower channel or trend turns bearish
            if (price < lower_channel[i] or
                price < ema_trend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Price breaks above upper channel or trend turns bullish
            if (price > upper_channel[i] or
                price > ema_trend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_1w_ATR_Breakout_Trend_Volume"
timeframe = "1d"
leverage = 1.0