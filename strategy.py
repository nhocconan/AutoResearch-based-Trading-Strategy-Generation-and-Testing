#!/usr/bin/env python3
# 1h_4h1d_Camarilla_R1_S1_Breakout_RSI_Momentum
# Hypothesis: 1h breakout above/below daily Camarilla R1/S1 with RSI momentum filter and volume confirmation.
# Uses 4h trend filter (price > 4h EMA50) for bias. Designed for low trade frequency (<30/year) to avoid fee drag.
# Works in bull/bear via 4h trend filter and momentum confirmation to avoid whipsaws.

name = "1h_4h1d_Camarilla_R1_S1_Breakout_RSI_Momentum"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Get daily data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema_50 = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50)
    
    # Daily high, low, close for Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla R1 = close + 1.1*(high-low)/12
    # Camarilla S1 = close - 1.1*(high-low)/12
    r1 = close_1d + 1.1 * (high_1d - low_1d) / 12
    s1 = close_1d - 1.1 * (high_1d - low_1d) / 12
    
    # Align R1 and S1 to 1h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # RSI(14) for momentum filter
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation (2.0x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_50_aligned[i]) or
            np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i]) or
            np.isnan(rsi[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter from 4h EMA50
        bullish_trend = close[i] > ema_50_aligned[i]
        bearish_trend = close[i] < ema_50_aligned[i]
        
        # Volume confirmation (2.0x average)
        volume_surge = volume[i] > 2.0 * vol_ma[i]
        
        # RSI momentum: avoid overbought/oversold extremes
        rsi_momentum_long = rsi[i] > 50 and rsi[i] < 70
        rsi_momentum_short = rsi[i] < 50 and rsi[i] > 30
        
        if position == 0:
            # Long: breakout above R1 in bullish trend with volume surge and RSI momentum
            if close[i] > r1_aligned[i] and bullish_trend and volume_surge and rsi_momentum_long:
                signals[i] = 0.20
                position = 1
            # Short: breakdown below S1 in bearish trend with volume surge and RSI momentum
            elif close[i] < s1_aligned[i] and bearish_trend and volume_surge and rsi_momentum_short:
                signals[i] = -0.20
                position = -1
        else:
            if position == 1:
                # Exit: close below R1 or RSI turns bearish
                if close[i] < r1_aligned[i] or rsi[i] < 40:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
            elif position == -1:
                # Exit: close above S1 or RSI turns bullish
                if close[i] > s1_aligned[i] or rsi[i] > 60:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
    
    return signals