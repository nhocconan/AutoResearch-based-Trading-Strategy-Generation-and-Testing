#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_RelativeStrengthIndex_Consensus_Trend"
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
    
    # 1-day RSI for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # 6-hour RSI for entry timing
    delta_6h = np.diff(close, prepend=close[0])
    gain_6h = np.where(delta_6h > 0, delta_6h, 0)
    loss_6h = np.where(delta_6h < 0, -delta_6h, 0)
    avg_gain_6h = pd.Series(gain_6h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss_6h = pd.Series(loss_6h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs_6h = avg_gain_6h / (avg_loss_6h + 1e-10)
    rsi_6h = 100 - (100 / (1 + rs_6h))
    
    # Volume filter: current volume > 1.2x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.2 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(rsi_1d_aligned[i]) or np.isnan(rsi_6h[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: 1d RSI > 50 (uptrend) + 6h RSI < 30 (oversold) + volume filter
            long_cond = (rsi_1d_aligned[i] > 50 and 
                        rsi_6h[i] < 30 and
                        volume_filter[i])
            
            # Short: 1d RSI < 50 (downtrend) + 6h RSI > 70 (overbought) + volume filter
            short_cond = (rsi_1d_aligned[i] < 50 and 
                         rsi_6h[i] > 70 and
                         volume_filter[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: 6h RSI crosses above 70 (overbought)
            if rsi_6h[i] >= 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: 6h RSI crosses below 30 (oversold)
            if rsi_6h[i] <= 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals