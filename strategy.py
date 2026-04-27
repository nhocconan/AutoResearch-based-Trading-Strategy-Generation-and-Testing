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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Weekly EMA trend filter (21-period)
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # Daily ATR for volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # 4h ATR for stoploss
    tr_4h1 = high[1:] - low[1:]
    tr_4h2 = np.abs(high[1:] - close[:-1])
    tr_4h3 = np.abs(low[1:] - close[:-1])
    tr_4h = np.concatenate([[np.nan], np.maximum(tr_4h1, np.maximum(tr_4h2, tr_4h3))])
    atr_14_4h = pd.Series(tr_4h).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need weekly EMA (21), daily ATR (14)
    start_idx = max(21, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_21_1w_aligned[i]) or np.isnan(atr_14_1d_aligned[i]) or
            np.isnan(atr_14_4h[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        atr_4h = atr_14_4h[i]
        
        # Trend filter from weekly EMA
        bullish_trend = price > ema_21_1w_aligned[i]
        bearish_trend = price < ema_21_1w_aligned[i]
        
        # Volatility filter: avoid extremely low volatility periods
        vol_filter = atr_14_1d_aligned[i] > 0.01 * price  # ATR > 1% of price
        
        if position == 0:
            # Long: price above weekly EMA + volatility filter
            if bullish_trend and vol_filter:
                signals[i] = size
                position = 1
            # Short: price below weekly EMA + volatility filter
            elif bearish_trend and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below weekly EMA or ATR-based stoploss
            if not bullish_trend or price < ema_21_1w_aligned[i] - 2.0 * atr_4h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above weekly EMA or ATR-based stoploss
            if not bearish_trend or price > ema_21_1w_aligned[i] + 2.0 * atr_4h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_WeeklyEMA21_Trend_VolumeFilter_ATRStop"
timeframe = "4h"
leverage = 1.0