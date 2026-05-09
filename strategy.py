#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h1d_TrendFollow_Volume"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend filter and volume average
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Calculate 4h EMA(34) for trend filter
    close_4h = pd.Series(df_4h['close'].values)
    ema34_4h = close_4h.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema34_4h)
    
    # Calculate 4h volume average (20-period)
    vol_4h = pd.Series(df_4h['volume'].values)
    vol_ma20_4h = vol_4h.rolling(window=20, min_periods=20).mean().values
    vol_ma20_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma20_4h)
    
    # Get 1d data for additional trend confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema50_1d = close_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Current volume for confirmation
    vol_series = pd.Series(volume)
    vol_ma20_current = vol_series.rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1h range for volatility filter
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    atr_period = 14
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    # Session filter: 8-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema34_4h_aligned[i]) or np.isnan(vol_ma20_4h_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma20_current[i]) or 
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Apply session filter
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: current volume > 1.5 * 4h average volume
        vol_ok = volume[i] > 1.5 * vol_ma20_4h_aligned[i]
        
        # Volatility filter: ATR > 0.5 * 20-period ATR average
        atr_ma20 = pd.Series(atr).rolling(window=20, min_periods=20).mean().values
        vol_filter = atr[i] > 0.5 * atr_ma20[i] if not np.isnan(atr_ma20[i]) else False
        
        if position == 0:
            # Long: Price above both 4h and 1d EMA with volume and volatility
            if (close[i] > ema34_4h_aligned[i] and 
                close[i] > ema50_1d_aligned[i] and 
                vol_ok and vol_filter):
                signals[i] = 0.20
                position = 1
            # Short: Price below both 4h and 1d EMA with volume and volatility
            elif (close[i] < ema34_4h_aligned[i] and 
                  close[i] < ema50_1d_aligned[i] and 
                  vol_ok and vol_filter):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: Price falls below 4h EMA or 1d EMA
            if close[i] < ema34_4h_aligned[i] or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: Price rises above 4h EMA or 1d EMA
            if close[i] > ema34_4h_aligned[i] or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals