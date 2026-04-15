#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h RSI mean reversion with 4h/1d trend filter and volume confirmation
# Targets 15-30 trades/year by combining RSI(14) extremes (<30/>70) with 4h EMA50 trend
# Uses 1d volume spike (>1.5x 20-period avg) to confirm institutional interest
# Works in bull markets (buy dips in uptrend) and bear markets (sell rallies in downtrend)
# Low frequency due to strict triple-condition requirement

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # RSI(14) calculation
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Load 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Load 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    vol_1d = df_1d['volume'].values
    vol_avg_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.20  # Fixed position size
    
    for i in range(14, n):
        # Skip if any required data is NaN
        if (np.isnan(rsi[i]) or np.isnan(ema50_4h_aligned[i]) or 
            np.isnan(vol_avg_aligned[i])):
            continue
            
        # Long: RSI oversold + uptrend + volume spike
        if (rsi[i] < 30 and 
            close[i] > ema50_4h_aligned[i] and 
            volume[i] > 1.5 * vol_avg_aligned[i] and 
            position <= 0):
            position = 1
            signals[i] = base_size
            
        # Short: RSI overbought + downtrend + volume spike
        elif (rsi[i] > 70 and 
              close[i] < ema50_4h_aligned[i] and 
              volume[i] > 1.5 * vol_avg_aligned[i] and 
              position >= 0):
            position = -1
            signals[i] = -base_size
            
        # Exit: RSI returns to neutral zone (40-60)
        elif position == 1 and rsi[i] >= 40:
            position = 0
            signals[i] = 0.0
        elif position == -1 and rsi[i] <= 60:
            position = 0
            signals[i] = 0.0
    
    return signals

name = "1h_RSI_4hTrend_1dVolume_MeanReversion"
timeframe = "1h"
leverage = 1.0