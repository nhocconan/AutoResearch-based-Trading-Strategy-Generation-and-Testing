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
    
    # Get 1d data for 12-period EMA (trend filter)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema12_1d = pd.Series(close_1d).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema12_6h = align_htf_to_ltf(prices, df_1d, ema12_1d)
    
    # Get 1d data for 14-period ATR (volatility filter)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]))
    tr2 = np.maximum(np.abs(low_1d[1:] - close_1d[:-1]), tr1)
    tr = np.concatenate([[np.nan], tr2])  # first TR is NaN
    atr14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr14_6h = align_htf_to_ltf(prices, df_1d, atr14_1d)
    
    # Get 1w data for 20-period EMA (long-term trend)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_6h = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # 6h ATR for position sizing (volatility scaling)
    tr_6h = np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1]))
    tr_6h = np.maximum(np.abs(low[1:] - close[:-1]), tr_6h)
    tr_6h = np.concatenate([[np.nan], tr_6h])
    atr14_6h_local = pd.Series(tr_6h).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # Need all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema12_6h[i]) or 
            np.isnan(atr14_6h[i]) or 
            np.isnan(ema20_6h[i]) or 
            np.isnan(atr14_6h_local[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: current 6h ATR > 1.2 * weekly ATR (avoid low vol chop)
        vol_filter = atr14_6h_local[i] > (1.2 * atr14_6h[i])
        
        # Trend alignment: price above both 12h EMA and weekly EMA
        price_above_both = close[i] > ema12_6h[i] and close[i] > ema20_6h[i]
        price_below_both = close[i] < ema12_6h[i] and close[i] < ema20_6h[i]
        
        if position == 0:
            # Long: price above both EMAs + volatility filter
            if price_above_both and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: price below both EMAs + volatility filter
            elif price_below_both and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below 12h EMA OR volatility drops
            if (close[i] < ema12_6h[i]) or (not vol_filter):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above 12h EMA OR volatility drops
            if (close[i] > ema12_6h[i]) or (not vol_filter):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_EMA12_EMA20_Vol_Filter"
timeframe = "6h"
leverage = 1.0