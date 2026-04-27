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
    
    # Get daily data for calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Daily close, high, low, volume
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    vol_1d = df_1d['volume'].values
    
    # 200-day SMA for trend filter
    sma_200 = pd.Series(close_1d).rolling(window=200, min_periods=200).mean().values
    
    # Daily ATR(14) for volatility
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 20-day average volume for volume filter
    vol_avg_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    
    # Get weekly data for trend confirmation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly EMA(50) for trend
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align daily indicators to 1d timeframe
    sma_200_aligned = align_htf_to_ltf(prices, df_1d, sma_200)
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    vol_avg_20_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need all indicators
    start_idx = max(200, 14, 20, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(sma_200_aligned[i]) or np.isnan(atr_14_aligned[i]) or 
            np.isnan(vol_avg_20_aligned[i]) or np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        sma_trend = sma_200_aligned[i]
        atr_val = atr_14_aligned[i]
        vol_avg = vol_avg_20_aligned[i]
        vol_current = volume[i]
        ema_trend = ema_50_1w_aligned[i]
        
        # Volatility filter: ATR > 20-period median (high volatility regime)
        if i >= 20:
            atr_ma = pd.Series(atr_14_aligned[:i+1]).rolling(window=20, min_periods=20).median().iloc[-1]
        else:
            atr_ma = atr_val
        vol_filter = atr_val > atr_ma
        
        # Volume filter: current volume > 1.5x daily average
        volume_filter = vol_current > (vol_avg * 1.5)
        
        if position == 0:
            # Long: price above 200-day SMA with weekly uptrend and filters
            if close[i] > sma_trend and close[i] > ema_trend and vol_filter and volume_filter:
                signals[i] = size
                position = 1
            # Short: price below 200-day SMA with weekly downtrend and filters
            elif close[i] < sma_trend and close[i] < ema_trend and vol_filter and volume_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below 200-day SMA or weekly trend turns down
            if close[i] < sma_trend or close[i] < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above 200-day SMA or weekly trend turns up
            if close[i] > sma_trend or close[i] > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_SMA200_WeeklyEMA50_ATR_Volume_Filter"
timeframe = "1d"
leverage = 1.0