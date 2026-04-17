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
    
    # Get daily data for ATR and EMA
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily ATR (14-period)
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = np.nan
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate daily EMA(34)
    ema_34_1d = pd.Series(close_1d).ewm(span=34, min_periods=34, adjust=False).mean().values
    
    # Align daily ATR and EMA to 6h timeframe
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 6h ATR for volatility filter
    tr1_6h = high - low
    tr2_6h = np.abs(high - np.roll(close, 1))
    tr3_6h = np.abs(low - np.roll(close, 1))
    tr1_6h[0] = np.nan
    tr2_6h[0] = np.nan
    tr3_6h[0] = np.nan
    tr_6h = np.maximum(tr1_6h, np.maximum(tr2_6h, tr3_6h))
    atr_6h = pd.Series(tr_6h).rolling(window=14, min_periods=14).mean().values
    
    # 6h EMA(20) for trend filter
    ema_20_6h = pd.Series(close).ewm(span=20, min_periods=20, adjust=False).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 20  # Need EMA20 and ATR14
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(atr_6h[i]) or 
            np.isnan(ema_20_6h[i]) or 
            np.isnan(atr_1d_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: 6h ATR > daily ATR (ensures sufficient volatility)
        volatility_filter = atr_6h[i] > atr_1d_aligned[i]
        
        # Trend filter: price above/below EMA20_6h
        price_above_ema = close[i] > ema_20_6h[i]
        price_below_ema = close[i] < ema_20_6h[i]
        
        # Regime filter: daily EMA34 slope (trending vs ranging)
        if i >= start_idx + 1:
            ema_slope = ema_34_1d_aligned[i] - ema_34_1d_aligned[i-1]
            trending_market = np.abs(ema_slope) > (0.1 * atr_1d_aligned[i])
        else:
            trending_market = True  # Default to allow trading until enough data
        
        if position == 0:
            # Long: price above EMA20, trending market, sufficient volatility
            if price_above_ema and trending_market and volatility_filter:
                signals[i] = 0.25
                position = 1
            # Short: price below EMA20, trending market, sufficient volatility
            elif price_below_ema and trending_market and volatility_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below EMA20 or volatility drops
            if close[i] < ema_20_6h[i] or not volatility_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above EMA20 or volatility drops
            if close[i] > ema_20_6h[i] or not volatility_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_EMA20_Trend_Filter_Volatility_Regime"
timeframe = "6h"
leverage = 1.0