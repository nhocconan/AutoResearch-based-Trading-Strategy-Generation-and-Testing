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
    
    # Get daily data for context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate daily ATR(14) for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_arr = df_1d['close'].values
    
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d_arr, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d_arr, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr14 = np.full(len(tr), np.nan)
    for i in range(13, len(tr)):
        atr14[i] = np.nanmean(tr[i-13:i+1])
    atr14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr14)
    
    # Calculate 12-period ATR for position sizing
    tr12 = np.abs(high - low)
    tr22 = np.abs(high - np.roll(close, 1))
    tr32 = np.abs(low - np.roll(close, 1))
    tr12[0] = tr22[0] = tr32[0] = np.nan
    tr12 = np.maximum(tr12, np.maximum(tr22, tr32))
    atr12 = np.full(n, np.nan)
    for i in range(11, n):
        atr12[i] = np.nanmean(tr12[i-11:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(atr14_1d_aligned[i]) or 
            np.isnan(atr12[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below daily EMA50
        price_above_ema50 = close[i] > ema50_1d_aligned[i]
        price_below_ema50 = close[i] < ema50_1d_aligned[i]
        
        # Volatility filter: current 12-period ATR > 1.5x daily ATR14 (elevated volatility)
        vol_filter = atr12[i] > atr14_1d_aligned[i] * 1.5
        
        # Entry conditions: volatility expansion in direction of trend
        long_entry = price_above_ema50 and vol_filter
        short_entry = price_below_ema50 and vol_filter
        
        # Exit conditions: trend reversal or volatility contraction
        long_exit = (close[i] < ema50_1d_aligned[i]) or (atr12[i] < atr14_1d_aligned[i] * 0.8)
        short_exit = (close[i] > ema50_1d_aligned[i]) or (atr12[i] < atr14_1d_aligned[i] * 0.8)
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_1d_ema50_vol_filter_v1"
timeframe = "12h"
leverage = 1.0