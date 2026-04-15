#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Mean Reversion with 4h Trend Filter and Volume Spike
# Uses 4h EMA for trend direction, 1h RSI for mean reversion signals, and volume spike for confirmation.
# Works in both bull and bear markets: long when price is below 4h EMA and oversold with volume spike,
# short when price is above 4h EMA and overbought with volume spike.
# Target: 60-150 total trades over 4 years = 15-37/year for 1h.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 4h data for EMA trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 21:
        return np.zeros(n)
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMA (21-period)
    ema_4h = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Calculate 1h RSI (14-period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.20  # Position size
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(rsi[i])):
            continue
        
        # Volume spike: current volume > 1.5 * 20-period median volume
        vol_median = np.median(volume[max(0, i-20):i+1]) if i >= 20 else np.median(volume[:i+1])
        volume_spike = volume[i] > 1.5 * vol_median
        
        # Long entry: price below 4h EMA, RSI oversold (<30), volume spike
        if (close[i] < ema_4h_aligned[i] and
            rsi[i] < 30 and
            volume_spike and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: price above 4h EMA, RSI overbought (>70), volume spike
        elif (close[i] > ema_4h_aligned[i] and
              rsi[i] > 70 and
              volume_spike and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: RSI returns to neutral range (40-60) or opposite signal
        elif position == 1 and (rsi[i] > 40 or close[i] > ema_4h_aligned[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (rsi[i] < 60 or close[i] < ema_4h_aligned[i]):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "1h_RSI_MeanReversion_4hEMA_Volume"
timeframe = "1h"
leverage = 1.0