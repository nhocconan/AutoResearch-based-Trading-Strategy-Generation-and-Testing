#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA(34) for trend
    close_12h_series = pd.Series(close_12h)
    ema_12h = close_12h_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Calculate 12h RSI(14) for overbought/oversold
    delta = pd.Series(close_12h).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_12h = 100 - (100 / (1 + rs))
    rsi_12h_aligned = align_htf_to_ltf(prices, df_12h, rsi_12h.values)
    
    # Volume confirmation: current volume > 1.5 * 4-period average
    volume_ma4 = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 34  # Need EMA and RSI
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_12h_aligned[i]) or 
            np.isnan(rsi_12h_aligned[i]) or 
            np.isnan(volume_ma4[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter
        volume_filter = volume[i] > (1.5 * volume_ma4[i])
        
        if position == 0:
            # Long: price above EMA and RSI < 30 (oversold) with volume
            if close[i] > ema_12h_aligned[i] and rsi_12h_aligned[i] < 30 and volume_filter:
                signals[i] = 0.25
                position = 1
            # Short: price below EMA and RSI > 70 (overbought) with volume
            elif close[i] < ema_12h_aligned[i] and rsi_12h_aligned[i] > 70 and volume_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below EMA or RSI > 70
            if close[i] < ema_12h_aligned[i] or rsi_12h_aligned[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above EMA or RSI < 30
            if close[i] > ema_12h_aligned[i] or rsi_12h_aligned[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_EMA34_RSI14_VolumeFilter"
timeframe = "6h"
leverage = 1.0