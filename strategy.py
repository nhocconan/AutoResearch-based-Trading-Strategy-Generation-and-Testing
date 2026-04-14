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
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly ATR(14) for volatility filter
    high_1w = pd.Series(df_1w['high'])
    low_1w = pd.Series(df_1w['low'])
    close_1w = pd.Series(df_1w['close'])
    tr1 = high_1w - low_1w
    tr2 = abs(high_1w - close_1w.shift(1))
    tr3 = abs(low_1w - close_1w.shift(1))
    tr_1w = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1w = tr_1w.rolling(window=14, min_periods=14).mean().values
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_1w)
    
    # Calculate weekly EMA(20) for trend filter
    ema_20_1w = close_1w.ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Calculate daily volume average for confirmation
    df_1d = get_htf_data(prices, '1d')
    vol_1d = pd.Series(df_1d['volume'])
    vol_avg_1d = vol_1d.rolling(window=20, min_periods=20).mean().values
    vol_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25
    
    # Start after enough data for calculations
    start = 100
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_20_1w_aligned[i]) or 
            np.isnan(atr_1w_aligned[i]) or 
            np.isnan(vol_avg_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        # Volatility filter: avoid extremely low volatility periods
        atr_ratio = atr_1w_aligned[i] / price if price > 0 else 0
        vol_filter = atr_ratio > 0.005  # Minimum 0.5% ATR relative to price
        
        # Volume confirmation: current volume > 2x daily average
        vol_confirm = vol > (vol_avg_1d_aligned[i] * 2.0) if not np.isnan(vol_avg_1d_aligned[i]) else False
        
        # Trend filter: price relative to weekly EMA20
        trend_filter_long = price > ema_20_1w_aligned[i]
        trend_filter_short = price < ema_20_1w_aligned[i]
        
        if position == 0:
            # Long setup: price above weekly EMA20 + volume confirmation + volatility filter
            if (trend_filter_long and vol_confirm and vol_filter):
                position = 1
                signals[i] = position_size
            # Short setup: price below weekly EMA20 + volume confirmation + volatility filter
            elif (trend_filter_short and vol_confirm and vol_filter):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below weekly EMA20
            if price < ema_20_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses above weekly EMA20
            if price > ema_20_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1wEMA20_Volume_Filter"
timeframe = "12h"
leverage = 1.0