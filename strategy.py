#!/usr/bin/env python3
# 6h_tdi_bollinger_volume_v1
# Hypothesis: Combines TDI (Traders Dynamic Index) with Bollinger Bands and volume confirmation on 6h timeframe.
# TDI = RSI smoothed + volatility bands. Buy when RSI crosses above lower band in oversold area with volume spike.
# Sell when RSI crosses below upper band in overbought area with volume spike.
# Uses 1d trend filter: only trade in direction of 1d EMA50 slope.
# Designed to work in both bull and bear markets by capturing mean reversion within the trend.
# Target: 60-120 total trades over 4 years (15-30/year).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_tdi_bollinger_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # TDI components: RSI(13) smoothed + Bollinger Bands
    rsi_period = 13
    rsi_ma_period = 2
    bb_period = 20
    bb_std = 2.0
    
    # Calculate RSI
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/rsi_period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/rsi_period, adjust=False).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Smooth RSI for TDI
    rsi_smoothed = pd.Series(rsi).ewm(span=rsi_ma_period, adjust=False).mean().values
    
    # Bollinger Bands on price
    close_series = pd.Series(close)
    bb_middle = close_series.rolling(window=bb_period, min_periods=bb_period).mean().values
    bb_std_dev = close_series.rolling(window=bb_period, min_periods=bb_period).std().values
    bb_upper = bb_middle + bb_std * bb_std_dev
    bb_lower = bb_middle - bb_std * bb_std_dev
    
    # Volume filter: 1.8x 20-period average
    vol_ma_period = 20
    vol_ma = np.full(n, np.nan)
    for i in range(vol_ma_period-1, n):
        vol_ma[i] = np.mean(volume[i-vol_ma_period+1:i+1])
    
    vol_surge = np.full(n, False)
    for i in range(n):
        if not np.isnan(vol_ma[i]) and vol_ma[i] > 0:
            vol_surge[i] = volume[i] > 1.8 * vol_ma[i]
    
    # Get 1d data for trend filter (EMA50 slope)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    # Calculate slope: positive if current EMA > EMA 3 periods ago
    ema50_slope_1d = np.full(len(close_1d), np.nan)
    for i in range(3, len(close_1d)):
        if not np.isnan(ema50_1d[i]) and not np.isnan(ema50_1d[i-3]):
            ema50_slope_1d[i] = ema50_1d[i] - ema50_1d[i-3]
    ema50_slope_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_slope_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = max(rsi_period, bb_period, vol_ma_period, 3) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(rsi_smoothed[i]) or np.isnan(bb_lower[i]) or np.isnan(bb_upper[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(ema50_slope_1d_aligned[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: RSI crosses below middle band or volume drops
            if rsi_smoothed[i] < bb_middle[i] or volume[i] < vol_ma[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: RSI crosses above middle band or volume drops
            if rsi_smoothed[i] > bb_middle[i] or volume[i] < vol_ma[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: RSI crosses above lower band in oversold (<30) with volume surge and 1d uptrend
            if (rsi_smoothed[i] > bb_lower[i] and 
                rsi_smoothed[i-1] <= bb_lower[i-1] and
                rsi_smoothed[i] < 30 and
                vol_surge[i] and
                ema50_slope_1d_aligned[i] > 0):
                position = 1
                signals[i] = 0.25
            # Short entry: RSI crosses below upper band in overbought (>70) with volume surge and 1d downtrend
            elif (rsi_smoothed[i] < bb_upper[i] and 
                  rsi_smoothed[i-1] >= bb_upper[i-1] and
                  rsi_smoothed[i] > 70 and
                  vol_surge[i] and
                  ema50_slope_1d_aligned[i] < 0):
                position = -1
                signals[i] = -0.25
    
    return signals