#!/usr/bin/env python3
name = "6h_RelativeStrength_Index_RSI_60_SMA_200_Trend_Filter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter and RSI calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # 1d RSI(60) for momentum - using 60-period for longer-term signal
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/60, adjust=False, min_periods=60).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/60, adjust=False, min_periods=60).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_60 = 100 - (100 / (1 + rs))
    rsi_60_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_60)
    
    # 1d SMA(200) for trend filter
    sma_200 = pd.Series(close_1d).rolling(window=200, min_periods=200).mean().values
    sma_200_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_200)
    
    # Volume filter: current volume > 1.3 * 20-period average (6h timeframe)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(200, 20)
    
    for i in range(start_idx, n):
        if np.isnan(rsi_60_1d_aligned[i]) or np.isnan(sma_200_1d_aligned[i]) or np.isnan(vol_avg[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: RSI > 60 (bullish momentum) + price above SMA200 (uptrend) + volume
            if rsi_60_1d_aligned[i] > 60 and close[i] > sma_200_1d_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: RSI < 40 (bearish momentum) + price below SMA200 (downtrend) + volume
            elif rsi_60_1d_aligned[i] < 40 and close[i] < sma_200_1d_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit: RSI crosses back to neutral zone (40-60) or trend reverses
            if position == 1:
                if rsi_60_1d_aligned[i] < 50 or close[i] < sma_200_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if rsi_60_1d_aligned[i] > 50 or close[i] > sma_200_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals