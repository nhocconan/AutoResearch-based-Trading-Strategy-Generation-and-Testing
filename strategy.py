#!/usr/bin/env python3
# 1d_1w_rsi_mean_reversion_volume_v1
# Hypothesis: Use 1w RSI for mean reversion signals in overbought/oversold conditions, 
# confirmed by 1d price action and volume spikes. Works in both bull and bear markets 
# by fading extremes during trends and catching reversals in ranges.
# Target: 15-25 trades/year per symbol (60-100 total over 4 years) via strict weekly RSI extremes.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_rsi_mean_reversion_volume_v1"
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
    
    # Get weekly data for RSI calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    # Calculate 14-period RSI on weekly close
    close_1w = df_1w['close'].values
    delta = np.diff(close_1w, prepend=close_1w[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing (equivalent to RMA)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1w = 100 - (100 / (1 + rs))
    
    # Align weekly RSI to daily timeframe (wait for weekly bar to close)
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    
    # Volume confirmation: volume > 2x average of last 20 days
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > vol_ma * 2.0
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not available
        if np.isnan(rsi_1w_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: RSI returns to neutral zone (50) or overbought (>70)
            if rsi_1w_aligned[i] >= 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
                
        elif position == -1:  # Short position
            # Exit: RSI returns to neutral zone (50) or oversold (<30)
            if rsi_1w_aligned[i] <= 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Long entry: RSI oversold (<30) with volume confirmation
            if rsi_1w_aligned[i] < 30 and vol_confirm[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: RSI overbought (>70) with volume confirmation
            elif rsi_1w_aligned[i] > 70 and vol_confirm[i]:
                position = -1
                signals[i] = -0.25
    
    return signals