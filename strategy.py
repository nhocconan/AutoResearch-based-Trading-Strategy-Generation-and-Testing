#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load weekly data for trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Weekly 50-period EMA for trend filter
    ema_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Daily data for ATR-based volatility filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Daily ATR (14-period) for volatility filter
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Daily close for momentum filter
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
    
    # ATR for position sizing and stop loss on 1d timeframe
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(60, n):
        # Skip if any data is not ready
        if (np.isnan(ema_1w_aligned[i]) or 
            np.isnan(atr_1d_aligned[i]) or 
            np.isnan(close_1d_aligned[i]) or 
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        ema_1w_val = ema_1w_aligned[i]
        atr_1d_val = atr_1d_aligned[i]
        close_1d_val = close_1d_aligned[i]
        atr_val = atr[i]
        
        if position == 0:
            # Long: price above weekly EMA50 (uptrend) + daily close above weekly EMA + low volatility
            if price > ema_1w_val and close_1d_val > ema_1w_val and atr_1d_val < np.nanpercentile(atr_1d_aligned[:i+1], 30):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price below weekly EMA50 (downtrend) + daily close below weekly EMA + low volatility
            elif price < ema_1w_val and close_1d_val < ema_1w_val and atr_1d_val < np.nanpercentile(atr_1d_aligned[:i+1], 30):
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position != 0:
            # Exit: trend reversal or volatility expansion
            trend_reversal = (position == 1 and close_1d_val < ema_1w_val) or \
                           (position == -1 and close_1d_val > ema_1w_val)
            vol_expansion = atr_1d_val > np.nanpercentile(atr_1d_aligned[:i+1], 70)
            
            if trend_reversal or vol_expansion:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_WeeklyEMA50_Trend_VolatilityFilter"
timeframe = "1d"
leverage = 1.0