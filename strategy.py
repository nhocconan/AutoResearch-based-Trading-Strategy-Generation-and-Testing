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
    
    # Get daily data for calculations (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Daily EMA34 for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema34_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Daily ATR(14) for volatility
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_arr = df_1d['close'].values
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d_arr, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d_arr, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr14_1d)
    
    # Weekly data for regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = pd.Series(df_1w['close'].values)
    ema50_1w = close_1w.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume filter: volume > 2.0x 30-period average (stricter to reduce trades)
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_filter = volume > (vol_ma * 2.0)
    
    # Volatility filter: ATR below its 50-period median (low volatility regime)
    atr_median = pd.Series(atr14_1d_aligned).rolling(window=50, min_periods=14).median().values
    vol_filter = atr14_1d_aligned < atr_median
    
    # Regime filter: price above weekly EMA50 = bullish regime
    bullish_regime = close > ema50_1w_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(atr14_1d_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr_median[i]) or np.isnan(ema50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above daily EMA34 + volume filter + low volatility + bullish regime
            if (close[i] > ema34_1d_aligned[i] and 
                volume_filter[i] and 
                vol_filter[i] and 
                bullish_regime[i]):
                signals[i] = 0.25
                position = 1
            # Short: price below daily EMA34 + volume filter + low volatility + bearish regime
            elif (close[i] < ema34_1d_aligned[i] and 
                  volume_filter[i] and 
                  vol_filter[i] and 
                  not bullish_regime[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price crosses below daily EMA34 (trend change) OR regime changes to bearish
            if close[i] < ema34_1d_aligned[i] or not bullish_regime[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above daily EMA34 (trend change) OR regime changes to bullish
            if close[i] > ema34_1d_aligned[i] or bullish_regime[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_EMA34_Vol_LowVol_Regime_v1"
timeframe = "12h"
leverage = 1.0