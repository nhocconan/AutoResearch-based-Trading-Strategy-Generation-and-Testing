#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for calculations
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate ATR(14) for volatility filtering
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Bollinger Bands (20, 2) for volatility regime
    close_series = pd.Series(close)
    sma20 = close_series.rolling(window=20, min_periods=20).mean().values
    std20 = close_series.rolling(window=20, min_periods=20).std().values
    upper_bb = sma20 + 2 * std20
    lower_bb = sma20 - 2 * std20
    bb_width = (upper_bb - lower_bb) / sma20
    
    # Bollinger Band width percentile for regime detection (50-period)
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=50, min_periods=50).rank(pct=True).values
    
    # Calculate daily EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(200, 50)  # 200 for BB width percentile, 50 for EMA
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(atr[i]) or np.isnan(bb_width_percentile[i]) or 
            np.isnan(ema50_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        # Only trade in low volatility regimes (BB width percentile < 30%)
        if bb_width_percentile[i] >= 0.3:
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
            
        if position == 0:
            # Long: price above EMA50 and bouncing off lower Bollinger Band
            if price > ema50_aligned[i] and price <= lower_bb[i] * 1.02:
                position = 1
                signals[i] = position_size
            # Short: price below EMA50 and bouncing off upper Bollinger Band
            elif price < ema50_aligned[i] and price >= upper_bb[i] * 0.98:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below EMA50 or reaches upper BB
            if price < ema50_aligned[i] or price >= upper_bb[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses above EMA50 or reaches lower BB
            if price > ema50_aligned[i] or price <= lower_bb[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_Bollinger_Bounce_EMA50Filter"
timeframe = "4h"
leverage = 1.0