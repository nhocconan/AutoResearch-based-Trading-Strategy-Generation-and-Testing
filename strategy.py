#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for higher timeframe context (1d)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate daily EMA(34) for trend direction
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate daily high/low for 20-day range
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    high_20d = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_20d = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    high_20d_aligned = align_htf_to_ltf(prices, df_1d, high_20d)
    low_20d_aligned = align_htf_to_ltf(prices, df_1d, low_20d)
    
    # Calculate daily ATR(14) for volatility filter
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # Get 4h data for entry timing
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Calculate 4h volume moving average for confirmation
    vol_ma_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    vol_ma_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_4h)
    
    # Precompute session filter (08-20 UTC)
    hours = prices.index.hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(high_20d_aligned[i]) or 
            np.isnan(low_20d_aligned[i]) or
            np.isnan(atr_14_aligned[i]) or
            np.isnan(vol_ma_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade during active hours
        if not session_mask[i]:
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below daily EMA34
        price_above_ema = close[i] > ema_34_1d_aligned[i]
        price_below_ema = close[i] < ema_34_1d_aligned[i]
        
        # Volatility filter: avoid extremely low volatility
        vol_filter = atr_14_aligned[i] > 0 and atr_14_aligned[i] < np.percentile(atr_14[~np.isnan(atr_14)], 80) if i > 20 else True
        
        # Volume filter: current 4h volume above average
        volume_filter = vol_ma_4h_aligned[i] > 0 and volume[i] > vol_ma_4h_aligned[i] * 0.7
        
        # Range position: where price is within the 20-day range
        range_position = (close[i] - low_20d_aligned[i]) / (high_20d_aligned[i] - low_20d_aligned[i] + 1e-10)
        
        # Long conditions: bullish trend + volume + not overbought + volatility OK
        long_condition = (price_above_ema and 
                         volume_filter and 
                         vol_filter and
                         range_position < 0.7)  # Not in upper 30% of range
        
        # Short conditions: bearish trend + volume + not oversold + volatility OK
        short_condition = (price_below_ema and 
                          volume_filter and 
                          vol_filter and
                          range_position > 0.3)  # Not in lower 30% of range
        
        if long_condition and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_condition and position >= 0:
            signals[i] = -0.25
            position = -1
        # Exit conditions: trend reversal or extreme range
        elif position == 1 and (not price_above_ema or range_position > 0.8):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (not price_below_ema or range_position < 0.2):
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_EMA34_20DRange_VolumeFilter"
timeframe = "4h"
leverage = 1.0