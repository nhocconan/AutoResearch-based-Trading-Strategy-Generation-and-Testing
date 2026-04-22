#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Daily close for 1d EMA trend
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 1d EMA34 for trend
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # 4h close for entry trigger
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 4h ATR for volatility and stop
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 4h EMA34 for dynamic support/resistance
    ema34_4h = pd.Series(close).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if any data is not ready
        if (np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(ema34_4h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        atr_val = atr[i]
        ema34_1d_val = ema34_1d_aligned[i]
        ema34_4h_val = ema34_4h[i]
        
        # Trend filter: only trade in direction of 1d EMA34
        long_trend = price > ema34_1d_val
        short_trend = price < ema34_1d_val
        
        if position == 0:
            # Long: price pulls back to 4h EMA34 in uptrend
            if long_trend and price <= ema34_4h_val + 0.5 * atr_val and price >= ema34_4h_val - 0.5 * atr_val:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price pulls back to 4h EMA34 in downtrend
            elif short_trend and price <= ema34_4h_val + 0.5 * atr_val and price >= ema34_4h_val - 0.5 * atr_val:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position != 0:
            # Exit: price moves 1.5*ATR away from EMA or contrary 1d trend
            adverse_move = (position == 1 and price < ema34_4h_val - 1.5 * atr_val) or \
                           (position == -1 and price > ema34_4h_val + 1.5 * atr_val)
            trend_fail = (position == 1 and price < ema34_1d_val) or \
                         (position == -1 and price > ema34_1d_val)
            
            if adverse_move or trend_fail:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_EMA34_Pullback_TrendFilter_v1"
timeframe = "4h"
leverage = 1.0