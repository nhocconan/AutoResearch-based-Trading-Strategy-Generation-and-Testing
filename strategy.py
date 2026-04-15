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
    
    # 4h ADX for trend strength (14-period)
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # True Range
    tr1 = np.maximum(high_4h[1:] - low_4h[1:], np.abs(high_4h[1:] - close_4h[:-1]))
    tr2 = np.maximum(np.abs(low_4h[1:] - close_4h[:-1]), tr1)
    tr_4h = np.concatenate([[np.nan], tr2])
    atr_4h = pd.Series(tr_4h).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Movement
    up_move = high_4h[1:] - high_4h[:-1]
    down_move = low_4h[:-1] - low_4h[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    # Smoothed
    plus_di = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr_4h
    minus_di = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr_4h
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx_4h = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    adx_4h_aligned = align_htf_to_ltf(prices, df_4h, adx_4h)
    
    # 1d RSI for overbought/oversold (14-period)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d = np.concatenate([[np.nan], rsi_1d])  # Align with original length
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # 1h RSI for entry timing (14-period)
    rsi_1h = pd.Series(close).ewm(span=14, adjust=False, min_periods=14).mean()
    rsi_1h = 100 - (100 / (1 + (rsi_1h / (pd.Series(close).ewm(span=14, adjust=False, min_periods=14).mean() + 1e-10))))
    rsi_1h = rsi_1h.values
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(adx_4h_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(rsi_1h[i]) or np.isnan(hours[i])):
            continue
        
        # Session filter: only trade 08-20 UTC
        if not (8 <= hours[i] <= 20):
            continue
        
        # Long: Strong uptrend (ADX>25) + 1d RSI not overbought (<70) + 1h RSI oversold (<30)
        if (adx_4h_aligned[i] > 25 and 
            rsi_1d_aligned[i] < 70 and 
            rsi_1h[i] < 30):
            signals[i] = 0.20
        
        # Short: Strong downtrend (ADX>25) + 1d RSI not oversold (>30) + 1h RSI overbought (>70)
        elif (adx_4h_aligned[i] > 25 and 
              rsi_1d_aligned[i] > 30 and 
              rsi_1h[i] > 70):
            signals[i] = -0.20
        
        # Exit: trend weakens (ADX<20) or RSI extremes
        elif (adx_4h_aligned[i] < 20 or 
              rsi_1d_aligned[i] > 80 or 
              rsi_1d_aligned[i] < 20):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "1h_ADX25_RSI14_1dFilter_Session08-20"
timeframe = "1h"
leverage = 1.0