#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Pre-compute hours for session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # Load 4h data once before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Load 1d data for higher timeframe context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 4h ATR for volatility filter
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    tr1 = np.abs(high_4h[1:] - low_4h[1:])
    tr2 = np.abs(high_4h[1:] - close_4h[:-1])
    tr3 = np.abs(low_4h[1:] - close_4h[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_4h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_4h)
    
    # 1d close for trend filter
    close_1d = df_1d['close'].values
    sma_50_1d = pd.Series(close_1d).rolling(window=50, min_periods=50).mean().values
    sma_50_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_50_1d)
    
    # 1h price action
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1h Bollinger Bands (20, 2)
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.20  # 20% position size
    
    for i in range(50, n):
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
        # Skip if any required data is not ready
        if (np.isnan(atr_4h_aligned[i]) or np.isnan(sma_50_1d_aligned[i]) or
            np.isnan(sma_20[i]) or np.isnan(std_20[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: avoid low volatility periods
        vol_filter = atr_4h_aligned[i] > 0
        
        # Trend filter: price vs 50-day SMA on 1d
        uptrend = close[i] > sma_50_1d_aligned[i]
        downtrend = close[i] < sma_50_1d_aligned[i]
        
        # Bollinger Band mean reversion signals
        touch_lower = low[i] <= lower_bb[i]
        touch_upper = high[i] >= upper_bb[i]
        
        if position == 0:
            if in_session and vol_filter:
                # Long when touching lower BB in uptrend
                if touch_lower and uptrend:
                    position = 1
                    signals[i] = position_size
                # Short when touching upper BB in downtrend
                elif touch_upper and downtrend:
                    position = -1
                    signals[i] = -position_size
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit when price touches upper BB or trend breaks
            if touch_upper or not uptrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit when price touches lower BB or trend breaks
            if touch_lower or not downtrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1h_4h1d_Bollinger_MeanReversion_TrendFilter_v1"
timeframe = "1h"
leverage = 1.0