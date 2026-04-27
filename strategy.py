#!/usr/bin/env python3
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
    
    # Get daily data for calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 10-period RSI on daily close (avoid look-ahead)
    close_series = pd.Series(df_1d['close'])
    delta = close_series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/10, adjust=False, min_periods=10).mean()
    avg_loss = loss.ewm(alpha=1/10, adjust=False, min_periods=10).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # RSI overbought/oversold levels
    rsi_overbought = 70
    rsi_oversold = 30
    
    # Calculate 20-period SMA on daily close for trend filter
    sma20_1d = close_series.rolling(window=20, min_periods=20).mean().values
    
    # Align daily indicators to 1d timeframe
    rsi_1d = align_htf_to_ltf(prices, df_1d, rsi)
    sma20_1d_aligned = align_htf_to_ltf(prices, df_1d, sma20_1d)
    
    # Volume spike detection (20-period average)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup period
    start_idx = max(20, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(rsi_1d[i]) or np.isnan(sma20_1d_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: RSI oversold with volume spike and price above SMA20
            if (rsi_1d[i] < rsi_oversold and volume_spike[i] and close[i] > sma20_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: RSI overbought with volume spike and price below SMA20
            elif (rsi_1d[i] > rsi_overbought and volume_spike[i] and close[i] < sma20_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: RSI returns to neutral or trend fails
            if (rsi_1d[i] >= 50 or close[i] < sma20_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: RSI returns to neutral or trend fails
            if (rsi_1d[i] <= 50 or close[i] > sma20_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_RSI10_MeanReversion_VolumeSpike"
timeframe = "1d"
leverage = 1.0