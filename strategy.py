#!/usr/bin/env python3
# 4H_3ATR_Breakout_Trend_Filter_Volume_Spike
# Hypothesis: 4-hour price breakout beyond 3x ATR from 20-period mean with daily trend filter and volume spike confirmation.
# Uses ATR-based volatility breakout to capture strong momentum moves in both bull and bear markets.
# Daily trend filter prevents counter-trend trades. Volume spike ensures momentum confirmation.
# Targets 20-40 trades/year to minimize fee drag. Uses discrete position sizing (0.25).

name = "4H_3ATR_Breakout_Trend_Filter_Volume_Spike"
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
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # Need enough data for EMA34
        return np.zeros(n)
    
    # Calculate daily EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 20-period mean price (average of high, low, close)
    price = (high + low + close) / 3.0
    price_mean = pd.Series(price).rolling(window=20, min_periods=20).mean().values
    
    # Calculate ATR (14-period) for volatility measurement
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Volatility filter: avoid low volatility periods (ATR < 0.2% of price)
    vol_filter = atr > 0.002 * close  # ATR > 0.2% of price
    
    # Volume filter: average volume (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)  # Ensure we have price mean and volume MA data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(price_mean[i]) or np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(atr[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0 or
            np.isnan(vol_filter[i]) or not vol_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: spike confirmation (1.8x average volume)
        volume_filter = volume[i] > 1.8 * vol_ma[i]
        
        # Calculate upper and lower breakout levels (mean ± 3*ATR)
        upper_breakout = price_mean[i] + 3.0 * atr[i]
        lower_breakout = price_mean[i] - 3.0 * atr[i]
        
        if position == 0:
            # Long: Price breaks above upper level + daily uptrend + volume spike
            if (close[i] > upper_breakout and 
                close[i] > ema_34_1d_aligned[i] and   # Daily uptrend filter
                volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below lower level + daily downtrend + volume spike
            elif (close[i] < lower_breakout and 
                  close[i] < ema_34_1d_aligned[i] and   # Daily downtrend filter
                  volume_filter):
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit: Price returns to the 20-period mean (mean reversion)
            if position == 1 and close[i] < price_mean[i]:
                signals[i] = 0.0
                position = 0
            elif position == -1 and close[i] > price_mean[i]:
                signals[i] = 0.0
                position = 0
            else:
                # Maintain position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals