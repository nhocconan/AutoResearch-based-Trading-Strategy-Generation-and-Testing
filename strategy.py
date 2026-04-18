#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily Williams %R with 1-day trend filter and volume confirmation on 12h timeframe.
# Uses daily Williams %R(14) to detect oversold/overbought conditions and daily EMA50 for trend direction.
# Enters on 12h breakouts above/below open with volume confirmation.
# Designed for fewer trades (target 20-50/year) to avoid fee drag in both bull and bear markets.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    open_price = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for Williams %R and EMA
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams %R(14) on daily data with proper min_periods
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    wr_14 = -100 * (highest_high_14 - close_1d) / (highest_high_14 - lowest_low_14 + 1e-10)
    
    # Align daily Williams %R14 to 12h timeframe
    wr_14_aligned = align_htf_to_ltf(prices, df_1d, wr_14)
    
    # Calculate EMA(50) on daily close
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align daily EMA50 to 12h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 12h ATR for stop loss
    tr_12h_1 = high - low
    tr_12h_2 = np.abs(high - np.roll(close, 1))
    tr_12h_3 = np.abs(low - np.roll(close, 1))
    tr_12h_1[0] = high[0] - low[0]
    tr_12h_2[0] = np.abs(high[0] - close[0])
    tr_12h_3[0] = np.abs(low[0] - close[0])
    tr_12h = np.maximum(tr_12h_1, np.maximum(tr_12h_2, tr_12h_3))
    atr_12h = pd.Series(tr_12h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Calculate volume moving average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # need daily EMA50, volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(wr_14_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr_12h[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        vol_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        # Trend filter: price above daily EMA50 (uptrend) or below (downtrend)
        trend_up = close[i] > ema_50_1d_aligned[i]
        trend_down = close[i] < ema_50_1d_aligned[i]
        
        if position == 0:
            # Long entry: Williams %R oversold (< -80) + price above 12h open + 0.3*ATR, with volume and trend filter
            if (wr_14_aligned[i] < -80 and 
                close[i] > open_price[i] + 0.3 * atr_12h[i] and 
                vol_confirmed and 
                trend_up):
                signals[i] = 0.25
                position = 1
            # Short entry: Williams %R overbought (> -20) + price below 12h open - 0.3*ATR, with volume and trend filter
            elif (wr_14_aligned[i] > -20 and 
                  close[i] < open_price[i] - 0.3 * atr_12h[i] and 
                  vol_confirmed and 
                  trend_down):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: Williams %R overbought (> -20) or ATR-based stop
            if wr_14_aligned[i] > -20 or close[i] < open_price[i] - 1.5 * atr_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Williams %R oversold (< -80) or ATR-based stop
            if wr_14_aligned[i] < -80 or close[i] > open_price[i] + 1.5 * atr_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsR14Daily_EMA50Daily_VolumeFilter"
timeframe = "12h"
leverage = 1.0