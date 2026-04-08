#!/usr/bin/env python3
# 1d_ema200_rsi14_volume_v1
# Hypothesis: On daily timeframe, use EMA200 trend filter with RSI14 for mean reversion entries and volume confirmation.
# Long when price is above EMA200, RSI < 30 (oversold), and volume > 1.5x average.
# Short when price is below EMA200, RSI > 70 (overbought), and volume > 1.5x average.
# Exit when RSI returns to neutral zone (40-60) or volume drops below average.
# Works in both bull and bear markets via EMA200 trend filter and RSI mean reversion.
# Target: 10-25 trades/year by using strict daily signals with volume confirmation.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_ema200_rsi14_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate EMA200 for trend filter
    ema200 = pd.Series(close).ewm(span=200, min_periods=200, adjust=False).mean().values
    
    # Calculate RSI14
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values  # Fill NaN with neutral 50
    
    # Volume confirmation: 20-period average
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 200
    
    for i in range(start_idx, n):
        # Skip if data not available
        if np.isnan(ema200[i]) or np.isnan(rsi[i]) or np.isnan(avg_volume[i]):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: RSI returns to neutral (40-60) or volume drops below average
            if rsi[i] >= 40 and rsi[i] <= 60 or volume[i] < avg_volume[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: RSI returns to neutral (40-60) or volume drops below average
            if rsi[i] >= 40 and rsi[i] <= 60 or volume[i] < avg_volume[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume confirmation: current volume > 1.5x average volume
            volume_ok = volume[i] > 1.5 * avg_volume[i]
            
            # Trend filter
            uptrend = close[i] > ema200[i]
            downtrend = close[i] < ema200[i]
            
            # Long entry: price above EMA200, RSI oversold (<30), volume confirmation
            if uptrend and rsi[i] < 30 and volume_ok:
                position = 1
                signals[i] = 0.25
            # Short entry: price below EMA200, RSI overbought (>70), volume confirmation
            elif downtrend and rsi[i] > 70 and volume_ok:
                position = -1
                signals[i] = -0.25
    
    return signals