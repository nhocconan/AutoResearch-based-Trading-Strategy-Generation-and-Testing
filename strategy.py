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
    
    # Get 4h data for higher timeframe context
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # 4h SMA 50 for trend direction
    sma_50_4h = pd.Series(close_4h).rolling(window=50, min_periods=50).mean().values
    sma_50_4h_aligned = align_htf_to_ltf(prices, df_4h, sma_50_4h)
    
    # 4h ATR for volatility filter
    tr_4h = np.maximum(
        high_4h[1:] - low_4h[1:],
        np.maximum(
            np.abs(high_4h[1:] - close_4h[:-1]),
            np.abs(low_4h[1:] - close_4h[:-1])
        )
    )
    tr_4h = np.concatenate([[np.nan], tr_4h])
    atr_4h = pd.Series(tr_4h).rolling(window=14, min_periods=14).mean().values
    atr_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_4h)
    
    # Price change for momentum
    price_change = np.diff(close, prepend=close[0])
    price_change_ma = pd.Series(price_change).rolling(window=10, min_periods=10).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(sma_50_4h_aligned[i]) or 
            np.isnan(atr_4h_aligned[i]) or np.isnan(price_change_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 4h SMA50
        price_above_sma = close[i] > sma_50_4h_aligned[i]
        price_below_sma = close[i] < sma_50_4h_aligned[i]
        
        # Momentum filter: price change aligned with trend
        mom_long = price_change_ma[i] > 0
        mom_short = price_change_ma[i] < 0
        
        # Volatility filter: avoid extremely low volatility
        vol_filter = atr_4h_aligned[i] > 0
        
        # Long conditions: price above SMA + positive momentum + volatility
        long_entry = price_above_sma and mom_long and vol_filter
        # Short conditions: price below SMA + negative momentum + volatility
        short_entry = price_below_sma and mom_short and vol_filter
        
        if long_entry and position != 1:
            signals[i] = 0.25
            position = 1
        elif short_entry and position != -1:
            signals[i] = -0.25
            position = -1
        # Exit conditions: trend reversal
        elif position == 1 and price_below_sma:
            signals[i] = 0.0
            position = 0
        elif position == -1 and price_above_sma:
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_SMA50_MomentumTrend_VolFilter"
timeframe = "4h"
leverage = 1.0