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
    
    # Get 4h data once for HTF context
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Get 1d data for additional context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 4h ATR(14) for volatility filtering
    tr = np.maximum(high_4h[1:] - low_4h[1:], 
                    np.maximum(np.abs(high_4h[1:] - close_4h[:-1]), 
                               np.abs(low_4h[1:] - close_4h[:-1])))
    tr = np.concatenate([[np.nan], tr])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 4h EMA(50) for trend
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 1d EMA(200) for long-term trend
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align 4h indicators to 1h timeframe
    atr_14_aligned = align_htf_to_ltf(prices, df_4h, atr_14)
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Volume filter: above average volume (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Hour filter: 8-20 UTC (most active trading hours)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_14_aligned[i]) or np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(ema_200_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 8-20 UTC
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            # Outside session: flatten position
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Volume filter: above average volume
        vol_filter = volume[i] > vol_ma[i]
        
        # Trend filters: 4h EMA50 and 1d EMA200 alignment
        uptrend = close[i] > ema_50_4h_aligned[i] and close[i] > ema_200_1d_aligned[i]
        downtrend = close[i] < ema_50_4h_aligned[i] and close[i] < ema_200_1d_aligned[i]
        
        # Volatility filter: ATR must be above minimum threshold
        vol_filter_atr = atr_14_aligned[i] > 0.001 * close[i]  # Avoid extremely low volatility periods
        
        # Entry conditions:
        # Long: price in uptrend with volume and volatility
        # Short: price in downtrend with volume and volatility
        long_entry = uptrend and vol_filter and vol_filter_atr
        short_entry = downtrend and vol_filter and vol_filter_atr
        
        # Exit conditions: trend reversal or loss of volume/volatility
        long_exit = not uptrend or not vol_filter or not vol_filter_atr
        short_exit = not downtrend or not vol_filter or not vol_filter_atr
        
        if long_entry and position <= 0:
            signals[i] = 0.20
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.20
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
    
    return signals

name = "1h_EMA50_200_Trend_Filter_Vol_Volatility"
timeframe = "1h"
leverage = 1.0