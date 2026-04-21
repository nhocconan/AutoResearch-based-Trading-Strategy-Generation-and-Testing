#!/usr/bin/env python3
"""
1d_MeanReversion_BB_Bounce_with_Volume
Hypothesis: Mean reversion on Bollinger Bands (20,2) at extremes (price touches upper/lower band) with volume confirmation (>1.5x 20-period average volume) and RSI filter (RSI<30 for long, RSI>70 for short). Uses 1w EMA50 trend filter to avoid counter-trend trades. Designed for low trade frequency (target 15-25 trades/year) to minimize fee drag while capturing reversals in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d and 1w HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 30 or len(df_1w) < 50:
        return np.zeros(n)
    
    # === 1d Bollinger Bands (20,2) ===
    close_1d = df_1d['close'].values
    sma_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper_band = sma_20 + 2 * std_20
    lower_band = sma_20 - 2 * std_20
    
    # === 1d RSI(14) ===
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.full_like(close_1d, np.nan)
    avg_loss = np.full_like(close_1d, np.nan)
    if len(close_1d) > 14:
        avg_gain[13] = np.nanmean(gain[1:14])
        avg_loss[13] = np.nanmean(loss[1:14])
        for i in range(14, len(close_1d)):
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # === 1d Volume Average (20-period) ===
    volume_1d = df_1d['volume'].values
    vol_avg_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # === 1w EMA50 trend filter ===
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Align 1d indicators to lower timeframe
    upper_band_aligned = align_htf_to_ltf(prices, df_1d, upper_band)
    lower_band_aligned = align_htf_to_ltf(prices, df_1d, lower_band)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    vol_avg_20_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup
        # Skip if indicators not ready
        if (np.isnan(upper_band_aligned[i]) or
            np.isnan(lower_band_aligned[i]) or
            np.isnan(rsi_aligned[i]) or
            np.isnan(vol_avg_20_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        volume_current = prices['volume'].iloc[i]
        upper = upper_band_aligned[i]
        lower = lower_band_aligned[i]
        rsi_val = rsi_aligned[i]
        vol_avg = vol_avg_20_aligned[i]
        ema_50_val = ema_50_1w_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume_current > 1.5 * vol_avg
        
        if position == 0:
            # Long: Price touches/lower band, RSI < 30 (oversold), volume confirmation, and above 1w EMA50 (uptrend bias)
            if (price_close <= lower and 
                rsi_val < 30 and 
                vol_confirm and 
                price_close > ema_50_val):
                signals[i] = 0.25
                position = 1
            # Short: Price touches/upper band, RSI > 70 (overbought), volume confirmation, and below 1w EMA50 (downtrend bias)
            elif (price_close >= upper and 
                  rsi_val > 70 and 
                  vol_confirm and 
                  price_close < ema_50_val):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit when price returns to middle (SMA20) or opposite band touch
            sma_20_aligned = align_htf_to_ltf(prices, df_1d, sma_20)
            if position == 1 and price_close >= sma_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            elif position == -1 and price_close <= sma_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_MeanReversion_BB_Bounce_with_Volume"
timeframe = "1d"
leverage = 1.0