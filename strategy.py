#!/usr/bin/env python3
"""
4h_12h_MeanReversion_Fade_V1
Hypothesis: Fade price extremes at 12h Bollinger Bands (2, 2) when 4h RSI shows extreme momentum,
but only in low-volatility (high mean-reversion) regimes. Uses Bollinger Band width percentile
to detect ranging markets where mean reversion works best. Designed for low trade frequency
(15-25/year) by requiring Bollinger Band touch + RSI extreme + regime filter.
Works in bull/bear via mean reversion logic and volatility regime filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_MeanReversion_Fade_V1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 12H BOLLINGER BANDS (2, 2) ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Bollinger Bands: 20-period SMA, 2 std dev
    sma_20 = np.zeros_like(close_12h)
    std_20 = np.zeros_like(close_12h)
    for i in range(len(close_12h)):
        if i < 19:
            sma_20[i] = np.nan
            std_20[i] = np.nan
        else:
            sma_20[i] = np.mean(close_12h[i-19:i+1])
            std_20[i] = np.std(close_12h[i-19:i+1])
    
    upper_band = sma_20 + (2 * std_20)
    lower_band = sma_20 - (2 * std_20)
    
    # === 12H BOLLINGER BAND WIDTH FOR REGIME DETECTION ===
    bb_width = (upper_band - lower_band) / sma_20
    # Percentile rank of BB width over 50 periods to detect low volatility (rangy) regime
    bb_width_percentile = np.full_like(bb_width, np.nan)
    for i in range(49, len(bb_width)):
        window = bb_width[i-49:i+1]
        if not np.all(np.isnan(window)):
            ranked = np.sum(~np.isnan(window) & (window < bb_width[i]))
            total = np.sum(~np.isnan(window))
            if total > 0:
                bb_width_percentile[i] = (ranked / total) * 100
    
    # Low volatility regime (rangy market) when BB width < 30th percentile
    low_vol_regime = bb_width_percentile < 30
    
    # === 4H RSI (14) FOR MOMENTUM EXTREMES ===
    # RSI calculation with proper handling
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    for i in range(len(gain)):
        if i < 13:
            avg_gain[i] = np.nan
            avg_loss[i] = np.nan
        elif i == 13:
            avg_gain[i] = np.mean(gain[0:14])
            avg_loss[i] = np.mean(loss[0:14])
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    
    # === ALIGN ALL 12H INDICATORS TO 4H TIMEFRAME ===
    upper_band_aligned = align_htf_to_ltf(prices, df_12h, upper_band)
    lower_band_aligned = align_htf_to_ltf(prices, df_12h, lower_band)
    low_vol_regime_aligned = align_htf_to_ltf(prices, df_12h, low_vol_regime.astype(float))
    
    # Volume average (20-period for 4h = ~5 days) for confirmation
    vol_avg = np.zeros(n)
    vol_sum = 0.0
    vol_count = 0
    for i in range(n):
        vol_sum += volume[i]
        vol_count += 1
        if i >= 20:
            vol_sum -= volume[i-20]
            vol_count -= 1
        if vol_count > 0:
            vol_avg[i] = vol_sum / vol_count
        else:
            vol_avg[i] = 0.0
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # start after warmup
        # Skip if indicators not available
        if (np.isnan(upper_band_aligned[i]) or np.isnan(lower_band_aligned[i]) or 
            np.isnan(low_vol_regime_aligned[i]) or np.isnan(rsi[i]) or vol_avg[i] == 0.0):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: at least 1.3x average
        vol_confirm = volume[i] > 1.3 * vol_avg[i]
        
        # Only trade in low volatility (rangy) regime where mean reversion works
        in_low_vol_regime = low_vol_regime_aligned[i] > 0.5
        
        # Fade extreme price action at Bollinger Bands with RSI confirmation
        # Long when price touches lower band AND RSI is oversold (< 30)
        long_setup = (close[i] <= lower_band_aligned[i]) and (rsi[i] < 30) and vol_confirm and in_low_vol_regime
        # Short when price touches upper band AND RSI is overbought (> 70)
        short_setup = (close[i] >= upper_band_aligned[i]) and (rsi[i] > 70) and vol_confirm and in_low_vol_regime
        
        # Exit when price returns to mean (SMA) or RSI normalizes
        exit_long = (close[i] >= sma_20[i]) or (rsi[i] >= 50)
        exit_short = (close[i] <= sma_20[i]) or (rsi[i] <= 50)
        
        if long_setup and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_setup and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals