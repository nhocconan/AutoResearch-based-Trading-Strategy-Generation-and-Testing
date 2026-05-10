#!/usr/bin/env python3
# 6h_1w_1d_Camarilla_R4_S4_Breakout_TrendVolume_v1
# Hypothesis: 6h breakout above weekly Camarilla R4/S4 with daily trend filter and volume confirmation.
# Weekly R4/S4 represent strong support/resistance; breakout with volume indicates institutional interest.
# Daily trend filter (EMA50) ensures alignment with higher timeframe momentum.
# Designed for low trade frequency (<30/year) to avoid fee drag in 6h timeframe.
# Works in bull/bear via daily trend filter and volatility-adjusted position sizing.

name = "6h_1w_1d_Camarilla_R4_S4_Breakout_TrendVolume_v1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get weekly data for Camarilla levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 6h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly high, low, close for Camarilla levels
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly Camarilla R4 = close + 1.1*(high-low)/2
    # Weekly Camarilla S4 = close - 1.1*(high-low)/2
    r4 = close_1w + 1.1 * (high_1w - low_1w) / 2
    s4 = close_1w - 1.1 * (high_1w - low_1w) / 2
    
    # Align R4 and S4 to 6h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1w, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1w, s4)
    
    # Daily EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Volume confirmation (2.5x 50-period average)
    vol_ma = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    
    # ATR for position sizing volatility adjustment
    tr1 = np.maximum(high - low, np.absolute(high - np.roll(close, 1)))
    tr2 = np.absolute(low - np.roll(close, 1))
    tr = np.maximum(tr1, tr2)
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(r4_aligned[i]) or
            np.isnan(s4_aligned[i]) or
            np.isnan(ema_50_aligned[i]) or
            np.isnan(vol_ma[i]) or
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter from daily EMA50
        bullish_trend = close[i] > ema_50_aligned[i]
        bearish_trend = close[i] < ema_50_aligned[i]
        
        # Volume confirmation (2.5x average)
        volume_surge = volume[i] > 2.5 * vol_ma[i]
        
        if position == 0:
            # Long: breakout above R4 in bullish trend with volume surge
            if close[i] > r4_aligned[i] and bullish_trend and volume_surge:
                signals[i] = 0.25
                position = 1
            # Short: breakdown below S4 in bearish trend with volume surge
            elif close[i] < s4_aligned[i] and bearish_trend and volume_surge:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: opposite breakout or loss of trend/volume
            if position == 1:
                # Exit long on breakdown below S4 or loss of bullish trend
                if close[i] < s4_aligned[i] or not bullish_trend or not volume_surge:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short on breakout above R4 or loss of bearish trend
                if close[i] > r4_aligned[i] or not bearish_trend or not volume_surge:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals