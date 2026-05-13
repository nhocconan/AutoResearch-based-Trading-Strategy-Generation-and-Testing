#!/usr/bin/env python3
# Hypothesis: 6h Bollinger Band Squeeze Breakout with 12h trend filter and 1d volume confirmation.
# Long when price breaks above upper BB(20,2) AND BB width < 20th percentile (squeeze) AND price > 12h EMA50 AND 1d volume > 1.3 * 20-period average volume.
# Short when price breaks below lower BB(20,2) AND BB width < 20th percentile AND price < 12h EMA50 AND 1d volume > 1.3 * 20-period average volume.
# Exit when price crosses back inside the Bollinger Bands (middle band).
# Uses discrete position sizing (0.25) to limit fee churn. Designed for BTC/ETH robustness by capturing low-volatility breakouts in trending markets with volume confirmation.
# Target: 60-100 total trades over 4 years (15-25/year) for 6h timeframe.

name = "6h_BollingerSqueezeBreakout_12hEMA50_1dVolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 12h EMA50 for trend filter (HTF)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 1d volume confirmation (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    volume_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume_1d > (1.3 * vol_ma_20)
    volume_confirm_aligned = align_htf_to_ltf(prices, df_1d, volume_confirm.astype(float))
    
    # Calculate Bollinger Bands (20,2) on primary timeframe
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_band = sma_20 + (2 * std_20)
    lower_band = sma_20 - (2 * std_20)
    bb_width = (upper_band - lower_band) / sma_20  # Normalized width
    
    # Calculate 20th percentile of BB width for squeeze condition (using expanding window to avoid look-ahead)
    bb_width_percentile = np.full(n, np.nan)
    for i in range(20, n):
        bb_width_percentile[i] = np.percentile(bb_width[20:i+1], 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after BB warmup
        # Skip if any required data is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(volume_confirm_aligned[i]) or
            np.isnan(bb_width[i]) or
            np.isnan(bb_width_percentile[i]) or
            np.isnan(sma_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above upper BB AND BB width < 20th percentile (squeeze) AND price > 12h EMA50 AND volume confirm
            if (close[i] > upper_band[i] and 
                bb_width[i] < bb_width_percentile[i] and 
                close[i] > ema_50_12h_aligned[i] and 
                volume_confirm_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below lower BB AND BB width < 20th percentile (squeeze) AND price < 12h EMA50 AND volume confirm
            elif (close[i] < lower_band[i] and 
                  bb_width[i] < bb_width_percentile[i] and 
                  close[i] < ema_50_12h_aligned[i] and 
                  volume_confirm_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses back inside BB (below upper band)
            if close[i] < upper_band[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses back inside BB (above lower band)
            if close[i] > lower_band[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals