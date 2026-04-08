#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_rsi_momentum"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for RSI momentum filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    # Calculate 14-period RSI on 1w close
    close_1w = df_1w['close'].values
    delta = np.diff(close_1w, prepend=close_1w[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1w = 100 - (100 / (1 + rs))
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    
    # Get 1d data for RSI and volume
    if len(prices) < 14:
        return np.zeros(n)
    
    # Calculate 14-period RSI on 1d close
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    
    # Volume confirmation: current volume > 1.5x 20-day average volume
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_avg_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup period
    start_idx = max(14, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if RSI not available
        if np.isnan(rsi_1w_aligned[i]) or np.isnan(rsi_1d[i]) or np.isnan(vol_avg_20[i]):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: RSI overbought or momentum weakens
            if rsi_1d[i] > 70 or rsi_1w_aligned[i] < 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
                
        elif position == -1:  # Short position
            # Exit: RSI oversold or momentum weakens
            if rsi_1d[i] < 30 or rsi_1w_aligned[i] > 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Long entry: RSI oversold on 1d, bullish momentum on 1w, with volume confirmation
            if (rsi_1d[i] < 30 and 
                rsi_1w_aligned[i] > 50 and 
                vol_confirm[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: RSI overbought on 1d, bearish momentum on 1w, with volume confirmation
            elif (rsi_1d[i] > 70 and 
                  rsi_1w_aligned[i] < 50 and 
                  vol_confirm[i]):
                position = -1
                signals[i] = -0.25
    
    return signals