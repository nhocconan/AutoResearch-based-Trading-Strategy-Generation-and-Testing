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
    
    # Get daily data for ATR and price
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily ATR(14) for volatility filter
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.concatenate([[high_1d[0] - low_1d[0]], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    atr_1d = np.full(len(df_1d), np.nan)
    for i in range(14, len(tr_1d)):
        atr_1d[i] = np.mean(tr_1d[i-14:i])
    
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Get weekly data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_1w_34 = np.full(len(df_1w), np.nan)
    alpha_w = 2 / (34 + 1)
    for i in range(len(close_1w)):
        if i == 0:
            ema_1w_34[i] = close_1w[i]
        elif i < 34:
            ema_1w_34[i] = np.mean(close_1w[:i+1])
        else:
            if np.isnan(ema_1w_34[i-1]):
                ema_1w_34[i] = np.mean(close_1w[i-33:i+1])
            else:
                ema_1w_34[i] = close_1w[i] * alpha_w + ema_1w_34[i-1] * (1 - alpha_w)
    
    ema_1w_34_aligned = align_htf_to_ltf(prices, df_1w, ema_1w_34)
    
    # Calculate 1-day ATR average (20-period) for volume filter
    atr_ma_20 = np.full(len(df_1d), np.nan)
    for i in range(20, len(atr_1d)):
        if not np.isnan(atr_1d[i]):
            atr_ma_20[i] = np.mean(atr_1d[i-20:i])
    
    atr_ma_20_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_20)
    
    signals = np.zeros(n)
    position = 0
    
    # Warmup: need all indicators
    start_idx = max(34, 20)  # weekly EMA needs 34, ATR MA needs 20
    
    for i in range(start_idx, n):
        if (np.isnan(atr_1d_aligned[i]) or
            np.isnan(atr_ma_20_aligned[i]) or
            np.isnan(ema_1w_34_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        atr_current = atr_1d_aligned[i]
        atr_ma = atr_ma_20_aligned[i]
        
        # Volatility filter: current ATR > 1.2x average ATR (avoid low volatility periods)
        volatility_filter = atr_current > (atr_ma * 1.2) if atr_ma > 0 else False
        
        if position == 0:
            # Long: price breaks above weekly EMA with volatility expansion
            if (volatility_filter and 
                price > ema_1w_34_aligned[i] and 
                close[i-1] <= ema_1w_34_aligned[i-1]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly EMA with volatility expansion
            elif (volatility_filter and 
                  price < ema_1w_34_aligned[i] and 
                  close[i-1] >= ema_1w_34_aligned[i-1]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price crosses below weekly EMA or volatility contracts
            if (price < ema_1w_34_aligned[i] or 
                atr_current < (atr_ma * 0.8)):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # Maintain position
        elif position == -1:
            # Short exit: price crosses above weekly EMA or volatility contracts
            if (price > ema_1w_34_aligned[i] or 
                atr_current < (atr_ma * 0.8)):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # Maintain position
    
    return signals

name = "1d_VolatilityExpansion_WeeklyEMA34_Trend_v1"
timeframe = "1d"
leverage = 1.0