#!/usr/bin/env python3
name = "6h_Volume_Weighted_RSI_Pullback_1dTrend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter and RSI calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 1d RSI(14) for overbought/oversold levels
    delta = pd.Series(close_1d).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi_values)
    
    # Calculate volume-weighted RSI(14) for 6s timeframe (more responsive to institutional flow)
    delta_6h = pd.Series(close).diff()
    gain_6h = delta_6h.clip(lower=0)
    loss_6h = -delta_6h.clip(upper=0)
    # Volume weighting: multiply gains/losses by volume
    vol_weighted_gain = (gain_6h * volume).ewm(span=14, adjust=False, min_periods=14).mean()
    vol_weighted_loss = (loss_6h * volume).ewm(span=14, adjust=False, min_periods=14).mean()
    vol_rs = vol_weighted_gain / vol_weighted_loss
    vol_rsi = 100 - (100 / (1 + vol_rs))
    vol_rsi_values = vol_rsi.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(rsi_aligned[i]) or 
            np.isnan(vol_rsi_values[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: 1d uptrend (price > EMA50), 6h vol-RSI oversold (<30), 1d RSI not overbought (<70)
            if (close[i] > ema_50_1d_aligned[i] and 
                vol_rsi_values[i] < 30 and 
                rsi_aligned[i] < 70):
                signals[i] = 0.25
                position = 1
            # Short: 1d downtrend (price < EMA50), 6h vol-RSI overbought (>70), 1d RSI not oversold (>30)
            elif (close[i] < ema_50_1d_aligned[i] and 
                  vol_rsi_values[i] > 70 and 
                  rsi_aligned[i] > 30):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: 6h vol-RSI overbought (>70) or 1d trend breaks down
            if vol_rsi_values[i] > 70 or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: 6h vol-RSI oversold (<30) or 1d trend breaks up
            if vol_rsi_values[i] < 30 or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals