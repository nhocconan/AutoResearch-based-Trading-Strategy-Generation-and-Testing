#!/usr/bin/env python3
name = "6h_RelativeStrengthIndex_Divergence_Volume"
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
    
    # Get 1d data for trend filter and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 1d RSI(14) for divergence detection
    delta = pd.Series(close_1d).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_values = rsi_1d.values
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d_values)
    
    # Volume filter: current 1d volume > 20-period average volume
    vol_1d = df_1d['volume'].values
    vol_avg = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    volume_filter = vol_1d > vol_avg
    volume_filter_aligned = align_htf_to_ltf(prices, df_1d, volume_filter)
    
    # Calculate 6h RSI(14) for entry timing
    delta_6h = pd.Series(close).diff()
    gain_6h = delta_6h.clip(lower=0)
    loss_6h = -delta_6h.clip(upper=0)
    avg_gain_6h = gain_6h.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss_6h = loss_6h.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs_6h = avg_gain_6h / avg_loss_6h
    rsi_6h = 100 - (100 / (1 + rs_6h))
    rsi_6h_values = rsi_6h.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(volume_filter_aligned[i]) or 
            np.isnan(rsi_6h_values[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: 1d uptrend, 6h RSI oversold (<30) with bullish divergence from 1d
            # Bullish divergence: price making lower low but RSI making higher low
            if (close[i] > ema_50_1d_aligned[i] and 
                rsi_6h_values[i] < 30 and 
                volume_filter_aligned[i]):
                # Check for bullish divergence: look back 3 periods for lower price low but higher RSI low
                if i >= 3:
                    price_low = min(close[i-2], close[i-1], close[i])
                    price_low_prev = min(close[i-5], close[i-4], close[i-3])
                    rsi_low = min(rsi_6h_values[i-2], rsi_6h_values[i-1], rsi_6h_values[i])
                    rsi_low_prev = min(rsi_6h_values[i-5], rsi_6h_values[i-4], rsi_6h_values[i-3])
                    if price_low < price_low_prev and rsi_low > rsi_low_prev:
                        signals[i] = 0.25
                        position = 1
            # Short: 1d downtrend, 6h RSI overbought (>70) with bearish divergence from 1d
            # Bearish divergence: price making higher high but RSI making lower high
            elif (close[i] < ema_50_1d_aligned[i] and 
                  rsi_6h_values[i] > 70 and 
                  volume_filter_aligned[i]):
                # Check for bearish divergence: look back 3 periods for higher price high but lower RSI high
                if i >= 3:
                    price_high = max(close[i-2], close[i-1], close[i])
                    price_high_prev = max(close[i-5], close[i-4], close[i-3])
                    rsi_high = max(rsi_6h_values[i-2], rsi_6h_values[i-1], rsi_6h_values[i])
                    rsi_high_prev = max(rsi_6h_values[i-5], rsi_6h_values[i-4], rsi_6h_values[i-3])
                    if price_high > price_high_prev and rsi_high < rsi_high_prev:
                        signals[i] = -0.25
                        position = -1
        elif position == 1:
            # Exit long: 6h RSI overbought (>70) or trend change
            if rsi_6h_values[i] > 70 or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: 6h RSI oversold (<30) or trend change
            if rsi_6h_values[i] < 30 or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals