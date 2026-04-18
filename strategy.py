#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for ATR-based volatility filter
    df_1d = get_htf_data(prices, '1d')
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range for daily ATR
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Daily ATR (14-period)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 4-hour EMA for trend direction (using 4h data)
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_21_4h = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all data to 4h timeframe
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    ema_21_aligned = align_htf_to_ltf(prices, df_4h, ema_21_4h)
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 14)  # need enough for EMA50 and ATR
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(atr_14_aligned[i]) or np.isnan(ema_21_aligned[i]) or 
            np.isnan(ema_50_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: EMA21 > EMA50 for uptrend, < for downtrend
        uptrend = ema_21_aligned[i] > ema_50_aligned[i]
        downtrend = ema_21_aligned[i] < ema_50_aligned[i]
        
        # Volatility filter: require sufficient volatility (ATR > 0)
        vol_filter = atr_14_aligned[i] > 0
        
        if position == 0:
            # Long: uptrend + volatility filter
            if uptrend and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: downtrend + volatility filter
            elif downtrend and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: trend reversal to downtrend
            if downtrend:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: trend reversal to uptrend
            if uptrend:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_EMA21_50_Trend_VolFilter"
timeframe = "4h"
leverage = 1.0