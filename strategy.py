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
    
    # Get weekly data for trend context (higher timeframe)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 20-period EMA on weekly for trend filter
    close_1w_series = pd.Series(close_1w)
    ema_20_1w = close_1w_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Get daily data for intermediate structure
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 14-period ATR on daily for volatility filter
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # Calculate 6-period ATR on 6h for entry sensitivity
    tr1_6h = np.abs(high[1:] - low[1:])
    tr2_6h = np.abs(high[1:] - close[:-1])
    tr3_6h = np.abs(low[1:] - close[:-1])
    tr_6h = np.concatenate([[np.nan], np.maximum(tr1_6h, np.maximum(tr2_6h, tr3_6h))])
    atr_6 = pd.Series(tr_6h).rolling(window=6, min_periods=6).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% of capital
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema_20_1w_aligned[i]) or 
            np.isnan(atr_14_aligned[i]) or 
            np.isnan(atr_6[i])):
            signals[i] = 0.0
            continue
        
        # Weekly trend filter: price above/below weekly EMA20
        above_weekly_ema = close[i] > ema_20_1w_aligned[i]
        below_weekly_ema = close[i] < ema_20_1w_aligned[i]
        
        # Volatility filter: avoid low volatility periods
        vol_filter = atr_14_aligned[i] > 0 and atr_6[i] > 0.5 * atr_14_aligned[i]
        
        # 6h momentum: price change over 3 periods
        if i >= 3:
            price_change = (close[i] - close[i-3]) / close[i-3]
        else:
            price_change = 0
        
        # Entry conditions: momentum in direction of weekly trend with volatility
        long_entry = (price_change > 0.01) and above_weekly_ema and vol_filter
        short_entry = (price_change < -0.01) and below_weekly_ema and vol_filter
        
        # Exit conditions: opposite momentum or trend reversal
        exit_long = position == 1 and (price_change < -0.005 or below_weekly_ema)
        exit_short = position == -1 and (price_change > 0.005 or above_weekly_ema)
        
        # Execute signals
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
        elif short_entry and position != -1:
            position = -1
            signals[i] = -position_size
        elif exit_long or exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_1w_ema20_momentum_vol_filter_v1"
timeframe = "6h"
leverage = 1.0