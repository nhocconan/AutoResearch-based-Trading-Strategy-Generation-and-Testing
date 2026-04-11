#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_1w_rsi_divergence_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 1d RSI(14)
    delta = np.diff(df_1d['close'].values)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_1d = np.where(avg_loss != 0, 100 - (100 / (1 + rs)), 100)
    # Pad with NaN for first 14 values
    rsi_1d = np.concatenate([np.full(14, np.nan), rsi_1d])
    
    # Calculate 1w RSI(14)
    delta_w = np.diff(df_1w['close'].values)
    gain_w = np.where(delta_w > 0, delta_w, 0)
    loss_w = np.where(delta_w < 0, -delta_w, 0)
    avg_gain_w = pd.Series(gain_w).rolling(window=14, min_periods=14).mean().values
    avg_loss_w = pd.Series(loss_w).rolling(window=14, min_periods=14).mean().values
    rs_w = np.where(avg_loss_w != 0, avg_gain_w / avg_loss_w, 0)
    rsi_1w = np.where(avg_loss_w != 0, 100 - (100 / (1 + rs_w)), 100)
    # Pad with NaN for first 14 values
    rsi_1w = np.concatenate([np.full(14, np.nan), rsi_1w])
    
    # Align 1d and 1w RSI to 6h timeframe
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    
    # Calculate 6h RSI(14) for divergence detection
    delta_6h = np.diff(close)
    gain_6h = np.where(delta_6h > 0, delta_6h, 0)
    loss_6h = np.where(delta_6h < 0, -delta_6h, 0)
    avg_gain_6h = pd.Series(gain_6h).rolling(window=14, min_periods=14).mean().values
    avg_loss_6h = pd.Series(loss_6h).rolling(window=14, min_periods=14).mean().values
    rs_6h = np.where(avg_loss_6h != 0, avg_gain_6h / avg_loss_6h, 0)
    rsi_6h = np.where(avg_loss_6h != 0, 100 - (100 / (1 + rs_6h)), 100)
    rsi_6h = np.concatenate([np.full(14, np.nan), rsi_6h])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start from index 100 to ensure sufficient data
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(rsi_1d_aligned[i]) or np.isnan(rsi_1w_aligned[i]) or 
            np.isnan(rsi_6h[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        rsi_1d_val = rsi_1d_aligned[i]
        rsi_1w_val = rsi_1w_aligned[i]
        rsi_6h_val = rsi_6h[i]
        
        # Bullish divergence: price makes lower low, RSI makes higher low
        # Bearish divergence: price makes higher high, RSI makes lower high
        lookback = 10  # Look back 10 periods for divergence
        
        if i >= lookback:
            # Price lows and highs
            price_low_6h = np.min(low[i-lookback:i+1])
            price_high_6h = np.max(high[i-lookback:i+1])
            price_low_prev = np.min(low[i-lookback*2:i-lookback+1])
            price_high_prev = np.max(high[i-lookback*2:i-lookback+1])
            
            # RSI lows and highs
            rsi_low_6h = np.nanmin(rsi_6h[i-lookback:i+1])
            rsi_high_6h = np.nanmax(rsi_6h[i-lookback:i+1])
            rsi_low_prev = np.nanmin(rsi_6h[i-lookback*2:i-lookback+1])
            rsi_high_prev = np.nanmax(rsi_6h[i-lookback*2:i-lookback+1])
            
            # Bullish divergence conditions
            bull_div = (price_low_6h < price_low_prev) and (rsi_low_6h > rsi_low_prev)
            # Bearish divergence conditions
            bear_div = (price_high_6h > price_high_prev) and (rsi_high_6h < rsi_high_prev)
            
            # Additional filters: RSI extremes and multi-timeframe alignment
            rsi_oversold = rsi_6h_val < 30
            rsi_overbought = rsi_6h_val > 70
            rsi_1d_oversold = rsi_1d_val < 30
            rsi_1d_overbought = rsi_1d_val > 70
            rsi_1w_oversold = rsi_1w_val < 30
            rsi_1w_overbought = rsi_1w_val > 70
            
            # Long signal: bullish divergence + oversold conditions
            long_signal = bull_div and (rsi_oversold or rsi_1d_oversold or rsi_1w_oversold)
            # Short signal: bearish divergence + overbought conditions
            short_signal = bear_div and (rsi_overbought or rsi_1d_overbought or rsi_1w_overbought)
            
            # Exit conditions: RSI returns to neutral territory
            exit_long = rsi_6h_val > 50
            exit_short = rsi_6h_val < 50
            
            if long_signal and position != 1:
                position = 1
                signals[i] = 0.25
            elif short_signal and position != -1:
                position = -1
                signals[i] = -0.25
            elif position == 1 and exit_long:
                position = 0
                signals[i] = 0.0
            elif position == -1 and exit_short:
                position = 0
                signals[i] = 0.0
            else:
                # Hold current position
                signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
        else:
            # Not enough data for divergence calculation
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
    
    return signals