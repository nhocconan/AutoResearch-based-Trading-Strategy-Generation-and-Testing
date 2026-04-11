#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_rsi_divergence_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop (primary timeframe)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Load 1w data ONCE before loop (higher timeframe)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate 14-period RSI on 1d closes
    delta = pd.Series(df_1d['close']).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Calculate 14-period RSI on 1w closes (for trend filter)
    delta_1w = pd.Series(df_1w['close']).diff()
    gain_1w = delta_1w.clip(lower=0)
    loss_1w = -delta_1w.clip(upper=0)
    avg_gain_1w = gain_1w.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss_1w = loss_1w.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs_1w = avg_gain_1w / avg_loss_1w
    rsi_1w = 100 - (100 / (1 + rs_1w))
    rsi_1w_values = rsi_1w.values
    
    # Align all indicators to 1d timeframe
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi_values)
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w_values)
    
    # Calculate 20-period SMA on 1w closes for trend filter
    sma_20_1w = pd.Series(df_1w['close']).rolling(window=20, min_periods=20).mean().values
    sma_20_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_20_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start from index 30 to ensure sufficient data
    for i in range(30, n):
        # Skip if any required data is invalid
        if (np.isnan(rsi_aligned[i]) or np.isnan(rsi_1w_aligned[i]) or 
            np.isnan(sma_20_1w_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        price = close[i]
        
        # RSI divergence conditions
        # Bearish divergence: price makes higher high, RSI makes lower high
        # Bullish divergence: price makes lower low, RSI makes higher low
        
        # Look back 3 periods for divergence detection
        if i >= 3:
            price_higher_high = (high[i] > high[i-1] and high[i-1] > high[i-2]) or \
                               (high[i] > high[i-2] and high[i-1] > high[i-3])
            price_lower_low = (low[i] < low[i-1] and low[i-1] < low[i-2]) or \
                             (low[i] < low[i-2] and low[i-1] < low[i-3])
            
            rsi_lower_high = (rsi_aligned[i] < rsi_aligned[i-1] and rsi_aligned[i-1] < rsi_aligned[i-2]) or \
                            (rsi_aligned[i] < rsi_aligned[i-2] and rsi_aligned[i-1] < rsi_aligned[i-3])
            rsi_higher_low = (rsi_aligned[i] > rsi_aligned[i-1] and rsi_aligned[i-1] > rsi_aligned[i-2]) or \
                            (rsi_aligned[i] > rsi_aligned[i-2] and rsi_aligned[i-1] > rsi_aligned[i-3])
        else:
            price_higher_high = False
            price_lower_low = False
            rsi_lower_high = False
            rsi_higher_low = False
        
        # Trend filter: 1w price above/below 20-period SMA
        uptrend = price > sma_20_1w_aligned[i]
        downtrend = price < sma_20_1w_aligned[i]
        
        # Long signal: bullish divergence + uptrend + RSI not overbought
        long_signal = (price_lower_low and rsi_higher_low and uptrend and 
                      rsi_aligned[i] < 70)
        
        # Short signal: bearish divergence + downtrend + RSI not oversold
        short_signal = (price_higher_high and rsi_lower_high and downtrend and 
                       rsi_aligned[i] > 30)
        
        # Exit conditions
        exit_long = rsi_aligned[i] > 70  # Overbought exit
        exit_short = rsi_aligned[i] < 30  # Oversold exit
        
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
    
    return signals