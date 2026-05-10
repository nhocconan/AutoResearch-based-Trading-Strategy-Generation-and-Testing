#!/usr/bin/env python3
# 4h_RangeBreakout_Volume_Strategy
# Hypothesis: Breakouts from Bollinger Band squeeze (low volatility) with volume confirmation and 1d trend filter.
# Works in bull/bear by trading breakouts in direction of daily trend. Bollinger Bands identify low volatility
# periods (squeeze) that precede explosive moves. Volume confirms breakout validity. Targets 20-50 trades/year.

name = "4h_RangeBreakout_Volume_Strategy"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Bollinger Bands (20, 2) - identifies volatility squeeze
    bb_period = 20
    bb_std = 2
    sma = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    std = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper_band = sma + (bb_std * std)
    lower_band = sma - (bb_std * std)
    bb_width = (upper_band - lower_band) / sma  # Normalized bandwidth
    
    # Bollinger Band squeeze detection (low volatility = potential breakout setup)
    # Squeeze when BB width is at lowest 20% of recent values
    bb_width_percentile = pd.Series(bb_width).rolling(window=50, min_periods=20).rank(pct=True).values
    squeeze_condition = bb_width_percentile < 0.2  # Bottom 20% = squeeze
    
    # Breakout detection: price breaks above/below Bollinger Bands
    breakout_up = close > upper_band
    breakout_down = close < lower_band
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > volume_ma * 1.5
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Daily EMA for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Daily trend filter
    uptrend = close > ema_34_1d_aligned
    downtrend = close < ema_34_1d_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(bb_period, 50, 20, 34)  # Warmup for BB, percentile, volume MA, daily EMA
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(sma[i]) or np.isnan(std[i]) or np.isnan(bb_width_percentile[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(ema_34_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: Bollinger Band breakout UP + squeeze + uptrend + volume
            if squeeze_condition[i] and breakout_up[i] and uptrend[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short entry: Bollinger Band breakout DOWN + squeeze + downtrend + volume
            elif squeeze_condition[i] and breakout_down[i] and downtrend[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price returns to middle Bollinger Band or trend reversal
            if close[i] <= sma[i] or not uptrend[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price returns to middle Bollinger Band or trend reversal
            if close[i] >= sma[i] or not downtrend[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals