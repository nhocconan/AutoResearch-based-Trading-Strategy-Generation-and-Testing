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
    
    # Get daily data for ATR and moving average
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily ATR(14)
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate daily EMA(50) for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 4-hour ATR(14) for stop loss
    tr1_4h = high - low
    tr2_4h = np.abs(high - np.roll(close, 1))
    tr3_4h = np.abs(low - np.roll(close, 1))
    tr_4h = np.maximum(tr1_4h, np.maximum(tr2_4h, tr3_4h))
    tr_4h[0] = tr1_4h[0]
    atr14_4h = pd.Series(tr_4h).rolling(window=14, min_periods=14).mean().values
    
    # Align daily indicators to 4h timeframe
    atr14_aligned = align_htf_to_ltf(prices, df_1d, atr14)
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate 4-hour moving average (20-period)
    ma20_4h = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(atr14_aligned[i]) or 
            np.isnan(ema50_aligned[i]) or
            np.isnan(ma20_4h[i]) or
            np.isnan(atr14_4h[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade during active hours
        if not session_mask[i]:
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below EMA50
        uptrend = close[i] > ema50_aligned[i]
        downtrend = close[i] < ema50_aligned[i]
        
        # Price position relative to 4h MA
        price_above_ma = close[i] > ma20_4h[i]
        price_below_ma = close[i] < ma20_4h[i]
        
        # Volatility filter: ATR above average (avoid low volatility periods)
        atr_ma = pd.Series(atr14_4h).rolling(window=20, min_periods=20).mean().values
        vol_filter = atr14_4h[i] > atr_ma[i] if not np.isnan(atr_ma[i]) else False
        
        # Entry conditions
        long_entry = uptrend and price_above_ma and vol_filter
        short_entry = downtrend and price_below_ma and vol_filter
        
        # Stop loss conditions
        long_stop = position == 1 and close[i] <= ma20_4h[i] - 2.0 * atr14_4h[i]
        short_stop = position == -1 and close[i] >= ma20_4h[i] + 2.0 * atr14_4h[i]
        
        # Exit conditions: trend reversal
        long_exit = position == 1 and (not uptrend or not price_above_ma)
        short_exit = position == -1 and (not downtrend or not price_below_ma)
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_stop or short_stop or long_exit or short_exit:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_ATR_EMA_MA_Trend_Filter"
timeframe = "4h"
leverage = 1.0