#!/usr/bin/env python3
# 4h_Hybrid_Technical_Signal_v1
# Hypothesis: Combine Bollinger Band squeeze breakout with 12h EMA trend filter and volume confirmation.
# Works in bull markets by catching breakouts from consolidation in uptrend.
# Works in bear markets by catching breakdowns from consolidation in downtrend.
# Uses Bollinger Band width to detect low volatility (squeeze) and price breakout for entry.
# 12h EMA filter ensures trades align with higher timeframe trend.
# Volume confirmation avoids false breakouts. Designed for low trade frequency (<50/year) to minimize fee drag.

name = "4h_Hybrid_Technical_Signal_v1"
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
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Bollinger Bands (20, 2) on 4h
    bb_period = 20
    bb_std = 2
    sma = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    std = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper_band = sma + (std * bb_std)
    lower_band = sma - (std * bb_std)
    
    # Bollinger Band Width for squeeze detection
    bb_width = (upper_band - lower_band) / sma
    # Squeeze when BB width is below its 50-period mean (low volatility)
    bb_width_ma = pd.Series(bb_width).rolling(window=50, min_periods=50).mean().values
    squeeze = bb_width < bb_width_ma
    
    # Volume confirmation (20-period MA)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need 12h EMA50 (50), BB (20), BB width MA (50), volume MA (20)
    start_idx = max(50, 20, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(sma[i]) or 
            np.isnan(std[i]) or 
            np.isnan(bb_width[i]) or 
            np.isnan(bb_width_ma[i]) or 
            np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter
        uptrend = close[i] > ema_50_12h_aligned[i]
        downtrend = close[i] < ema_50_12h_aligned[i]
        
        # Volume confirmation (>1.5x MA to balance sensitivity and noise)
        volume_confirm = volume[i] > volume_ma[i] * 1.5
        
        if position == 0:
            # Long entry: squeeze breakout above upper band + uptrend + volume
            if squeeze[i] and close[i] > upper_band[i] and uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: squeeze breakout below lower band + downtrend + volume
            elif squeeze[i] and close[i] < lower_band[i] and downtrend and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: trend breakdown or price returns below SMA (mean reversion)
            if not uptrend or close[i] < sma[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: trend reversal or price returns above SMA
            if not downtrend or close[i] > sma[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals