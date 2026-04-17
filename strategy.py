#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # 4h EMA50 for trend filter
    close_4h_series = pd.Series(close_4h)
    ema50_4h = close_4h_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Get daily data for ATR-based volatility filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Daily ATR(14) for volatility filter
    high_low_1d = high_1d - low_1d
    high_close_1d = np.abs(high_1d - np.roll(close_1d, 1))
    low_close_1d = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(high_low_1d, np.maximum(high_close_1d, low_close_1d))
    tr_1d[0] = high_low_1d[0]
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # 1h Bollinger Bands for entry timing
    close_series = pd.Series(close)
    bb_middle = close_series.rolling(window=20, min_periods=20).mean().values
    bb_std = close_series.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema50_4h_aligned[i]) or np.isnan(atr_1d_aligned[i]) or 
            np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Apply session filter
        if not session_filter[i]:
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price touches lower Bollinger Band in uptrend with volume
            if close[i] <= bb_lower[i] and close[i] > ema50_4h_aligned[i] and volume_filter[i]:
                signals[i] = 0.20
                position = 1
            # Short: price touches upper Bollinger Band in downtrend with volume
            elif close[i] >= bb_upper[i] and close[i] < ema50_4h_aligned[i] and volume_filter[i]:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: price touches middle Bollinger Band or volatility filter fails
            if close[i] >= bb_middle[i] or atr_1d_aligned[i] < (pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().iloc[i] * 0.5 if i >= 14 else 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: price touches middle Bollinger Band or volatility filter fails
            if close[i] <= bb_middle[i] or atr_1d_aligned[i] < (pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().iloc[i] * 0.5 if i >= 14 else 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_BollingerBandTouch_4hEMA50_Trend_Volume_Session"
timeframe = "1h"
leverage = 1.0