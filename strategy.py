#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1-week RSI extremes with 1-day ADX filter and volume confirmation
# Long when weekly RSI < 30 (oversold) + daily ADX > 25 (trending) + volume > 1.5x average
# Short when weekly RSI > 70 (overbought) + daily ADX > 25 (trending) + volume > 1.5x average
# Weekly RSI identifies extremes, daily ADX ensures trending environment, volume confirms strength.
# Works in bull/bear markets by fading extremes in trending conditions.
# Target: 12-37 trades per year (50-150 over 4 years) with 0.25 position sizing.

name = "6h_1wRSI_1dADX_TrendFade_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1-week RSI ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 14:
        return np.zeros(n)
    
    # Weekly RSI (14-period)
    delta = pd.Series(df_1w['close']).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_1w = (100 - (100 / (1 + rs))).fillna(50).values
    
    # Align weekly RSI to 6h timeframe
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    
    # Calculate 1-day ADX for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # True Range calculation for 1-day ADX
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    dm_plus = np.where((df_1d['high'] - df_1d['high'].shift(1)) > (df_1d['low'].shift(1) - df_1d['low']), 
                       np.maximum(df_1d['high'] - df_1d['high'].shift(1), 0), 0)
    dm_minus = np.where((df_1d['low'].shift(1) - df_1d['low']) > (df_1d['high'] - df_1d['high'].shift(1)), 
                        np.maximum(df_1d['low'].shift(1) - df_1d['low'], 0), 0)
    
    # Smooth TR and DM
    tr_smooth = pd.Series(tr).rolling(window=14, min_periods=14).sum()
    dm_plus_smooth = pd.Series(dm_plus).rolling(window=14, min_periods=14).sum()
    dm_minus_smooth = pd.Series(dm_minus).rolling(window=14, min_periods=14).sum()
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / tr_smooth.replace(0, np.nan)
    di_minus = 100 * dm_minus_smooth / tr_smooth.replace(0, np.nan)
    
    # DX and ADX
    dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus).replace(0, np.nan)
    adx_1d = dx.rolling(window=14, min_periods=14).mean().values
    
    # Align ADX to 6h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Volume confirmation: >1.5x 50-period average
    vol_ma_50 = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    volume_filter = volume > (1.5 * vol_ma_50)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(rsi_1w_aligned[i]) or np.isnan(adx_1d_aligned[i]) or 
            np.isnan(volume_filter[i]) or not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: weekly RSI oversold (<30) + trending (ADX>25) + volume confirmation
            if rsi_1w_aligned[i] < 30 and adx_1d_aligned[i] > 25 and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: weekly RSI overbought (>70) + trending (ADX>25) + volume confirmation
            elif rsi_1w_aligned[i] > 70 and adx_1d_aligned[i] > 25 and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: weekly RSI returns to neutral (>50) or ADX weakens (<20)
            if rsi_1w_aligned[i] > 50 or adx_1d_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: weekly RSI returns to neutral (<50) or ADX weakens (<20)
            if rsi_1w_aligned[i] < 50 or adx_1d_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals