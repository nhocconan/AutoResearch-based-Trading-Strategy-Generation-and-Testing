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
    
    # Get daily data for ATR and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily ATR(14)
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.concatenate([[high_1d[0] - low_1d[0]], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    atr_1d = np.full(len(tr_1d), np.nan)
    for i in range(14, len(tr_1d)):
        atr_1d[i] = np.mean(tr_1d[i-14:i])
    
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate daily EMA(50) for trend filter
    close_1d_series = pd.Series(close_1d)
    ema_1d_50 = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_50_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_50)
    
    # Calculate 60-period ATR for volatility filter (on 6h data)
    tr6h_1 = high[1:] - low[1:]
    tr6h_2 = np.abs(high[1:] - close[:-1])
    tr6h_3 = np.abs(low[1:] - close[:-1])
    tr_6h = np.concatenate([[high[0] - low[0]], np.maximum(tr6h_1, np.maximum(tr6h_2, tr6h_3))])
    
    atr_60 = np.full(n, np.nan)
    for i in range(60, n):
        atr_60[i] = np.mean(tr_6h[i-60:i])
    
    # Calculate Bollinger Bands (20, 2.0) on 6h close
    close_series = pd.Series(close)
    bb_mid = close_series.rolling(window=20, min_periods=20).mean().values
    bb_std = close_series.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_mid + 2.0 * bb_std
    bb_lower = bb_mid - 2.0 * bb_std
    
    signals = np.zeros(n)
    position = 0
    
    # Warmup: need all indicators
    start_idx = max(60, 50)  # ATR60 needs 60, EMA50 needs 50
    
    for i in range(start_idx, n):
        if (np.isnan(atr_1d_aligned[i]) or
            np.isnan(ema_1d_50_aligned[i]) or
            np.isnan(atr_60[i]) or
            np.isnan(bb_upper[i]) or
            np.isnan(bb_lower[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_factor = atr_60[i] / atr_1d_aligned[i] if atr_1d_aligned[i] > 0 else 0
        
        # Volatility filter: trade only when 6h ATR is between 0.5x and 2.0x daily ATR
        vol_filter = (vol_factor > 0.5) and (vol_factor < 2.0)
        
        if position == 0:
            # Long: price touches lower BB with low volatility and uptrend
            if (vol_filter and 
                price <= bb_lower[i] and 
                close[i-1] > bb_lower[i] and  # just touched
                close[i] > ema_1d_50_aligned[i]):  # above daily EMA50 (uptrend)
                signals[i] = 0.25
                position = 1
            # Short: price touches upper BB with low volatility and downtrend
            elif (vol_filter and 
                  price >= bb_upper[i] and 
                  close[i-1] < bb_upper[i] and  # just touched
                  close[i] < ema_1d_50_aligned[i]):  # below daily EMA50 (downtrend)
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price reaches middle BB or volatility expands too much
            if (price >= bb_mid[i] or 
                vol_factor > 2.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # Maintain position
        elif position == -1:
            # Short exit: price reaches middle BB or volatility expands too much
            if (price <= bb_mid[i] or 
                vol_factor > 2.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # Maintain position
    
    return signals

name = "6h_BollingerBands_ATRVolatilityFilter_DailyEMA50"
timeframe = "6h"
leverage = 1.0