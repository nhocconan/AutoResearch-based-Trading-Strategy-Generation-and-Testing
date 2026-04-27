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
    
    # Get 4h data for higher timeframe context
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    volume_4h = df_4h['volume'].values
    
    # Calculate 4h SMA(20) for trend direction
    sma_20_4h = pd.Series(close_4h := pd.Series(high_4h).rolling(window=20, min_periods=20).mean().values).rolling(window=20, min_periods=20).mean().values
    sma_20_4h_aligned = align_htf_to_ltf(prices, df_4h, sma_20_4h)
    
    # Calculate 4h volume moving average for confirmation
    vol_ma_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    vol_ma_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_4h)
    
    # Calculate 1h RSI(14) for entry timing
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Precompute session filter (08-20 UTC)
    hours = prices.index.hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(sma_20_4h_aligned[i]) or 
            np.isnan(vol_ma_4h_aligned[i]) or
            np.isnan(rsi[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade during active hours
        if not session_mask[i]:
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 4h SMA20
        price_above_sma = close[i] > sma_20_4h_aligned[i]
        price_below_sma = close[i] < sma_20_4h_aligned[i]
        
        # Volume filter: current 4h volume above average
        volume_filter = vol_ma_4h_aligned[i] > 0 and volume[i] > vol_ma_4h_aligned[i] * 1.2
        
        # RSI filters: avoid extreme overbought/oversold
        rsi_not_overbought = rsi[i] < 70
        rsi_not_oversold = rsi[i] > 30
        
        # Long conditions: bullish trend + volume + RSI not overbought
        long_condition = (price_above_sma and 
                         volume_filter and 
                         rsi_not_overbought)
        
        # Short conditions: bearish trend + volume + RSI not oversold
        short_condition = (price_below_sma and 
                          volume_filter and 
                          rsi_not_oversold)
        
        if long_condition and position <= 0:
            signals[i] = 0.20
            position = 1
        elif short_condition and position >= 0:
            signals[i] = -0.20
            position = -1
        # Exit conditions: trend reversal
        elif position == 1 and not price_above_sma:
            signals[i] = 0.0
            position = 0
        elif position == -1 and not price_below_sma:
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

name = "1h_4hSMA20_VolumeFilter_RSI14"
timeframe = "1h"
leverage = 1.0