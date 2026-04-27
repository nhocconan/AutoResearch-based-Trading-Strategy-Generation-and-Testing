#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h momentum with 4h trend filter and volume confirmation for balanced performance in bull/bear.
# Long when RSI(14) < 30 (oversold) and price > 4h EMA50 and volume > 1.5x average.
# Short when RSI(14) > 70 (overbought) and price < 4h EMA50 and volume > 1.5x average.
# Exit when RSI returns to neutral zone (40-60) to avoid whipsaw.
# Uses 4h EMA50 for trend filter to align with higher timeframe bias.
# Target: 60-150 total trades over 4 years (~15-37/year) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # 4-hour EMA50 for trend filter
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # RSI(14) calculation
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema50_4h_aligned[i]) or np.isnan(rsi_values[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long condition: RSI oversold, above 4h EMA50, volume spike
        if (rsi_values[i] < 30 and 
            close[i] > ema50_4h_aligned[i] and 
            volume_filter[i]):
            signals[i] = 0.20
            position = 1
        # Short condition: RSI overbought, below 4h EMA50, volume spike
        elif (rsi_values[i] > 70 and 
              close[i] < ema50_4h_aligned[i] and 
              volume_filter[i]):
            signals[i] = -0.20
            position = -1
        # Exit conditions: RSI returns to neutral zone (40-60)
        elif position == 1 and rsi_values[i] > 40:
            signals[i] = 0.0
            position = 0
        elif position == -1 and rsi_values[i] < 60:
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
    
    return signals

name = "1h_RSI14_4hEMA50_VolumeFilter"
timeframe = "1h"
leverage = 1.0