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
    
    # Get daily data for trend and volatility
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Daily EMA50 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Daily ATR(14) for volatility filter and stop
    high_low = high_1d - low_1d
    high_close = np.abs(high_1d - np.roll(close_1d, 1))
    low_close = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(high_low, np.maximum(high_close, low_close))
    tr_1d[0] = high_low[0]
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # 4h ATR(14) for position sizing and stop
    high_low = high - low
    high_close = np.abs(high - np.roll(close, 1))
    low_close = np.abs(low - np.roll(close, 1))
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    tr[0] = high_low[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 4-hour Bollinger Bands for mean reversion signals
    close_s = pd.Series(close)
    bb_mid = close_s.rolling(window=20, min_periods=20).mean().values
    bb_std = close_s.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_mid + (2.0 * bb_std)
    bb_lower = bb_mid - (2.0 * bb_std)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(atr_1d_aligned[i]) or 
            np.isnan(atr[i]) or np.isnan(bb_mid[i]) or np.isnan(bb_upper[i]) or 
            np.isnan(bb_lower[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price below lower BB in uptrend (daily EMA50) with volatility filter
            if close[i] < bb_lower[i] and close[i] > ema50_1d_aligned[i] and atr[i] > 0.5 * atr_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price above upper BB in downtrend (daily EMA50) with volatility filter
            elif close[i] > bb_upper[i] and close[i] < ema50_1d_aligned[i] and atr[i] > 0.5 * atr_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses above middle BB OR ATR-based stop
            if close[i] > bb_mid[i] or close[i] < (high[max(0, i-1)] - 1.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses below middle BB OR ATR-based stop
            if close[i] < bb_mid[i] or close[i] > (low[max(0, i-1)] + 1.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_BollingerMeanReversion_DailyTrend_VolatilityFilter"
timeframe = "4h"
leverage = 1.0