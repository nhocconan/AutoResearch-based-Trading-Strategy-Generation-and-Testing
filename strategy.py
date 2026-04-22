#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load daily data for weekly EMA (HTF) once
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Weekly EMA (50) on daily data
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Daily ATR for volatility regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    # Daily price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Daily ATR for entry trigger and stop
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Daily close for trend direction
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any data is not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(atr_1d[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(sma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        atr_val = atr[i]
        ema_50_1d_val = ema_50_1d_aligned[i]
        sma_20_val = sma_20[i]
        atr_1d_val = atr_1d[i]
        
        # Volatility regime: only trade when daily ATR is elevated (trending market)
        vol_regime = atr_1d_val > np.nanmedian(atr_1d[max(0, i-50):i+1])
        
        if position == 0 and vol_regime:
            # Long: price above weekly EMA50 and breaks above 20-day SMA + 1.5*ATR
            if price > ema_50_1d_val and price > sma_20_val + 1.5 * atr_val:
                signals[i] = 0.25
                position = 1
            # Short: price below weekly EMA50 and breaks below 20-day SMA - 1.5*ATR
            elif price < ema_50_1d_val and price < sma_20_val - 1.5 * atr_val:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: mean reversion to 20-day SMA or volatility collapse
            mean_rev = (position == 1 and price < sma_20_val) or (position == -1 and price > sma_20_val)
            vol_collapse = atr_val < 0.5 * atr[i-1] if i > 0 else False
            
            if mean_rev or vol_collapse:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_WeeklyEMA50_Trend_ATRBreakout_v1"
timeframe = "1d"
leverage = 1.0