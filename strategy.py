#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_21ema_Trend_Breakout_With_Confirmation"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # === 1d: Calculate 21 EMA for trend filter ===
    close_1d = df_1d['close'].values
    ema21_1d = pd.Series(close_1d).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema21_1d_aligned = align_htf_to_ltf(prices, df_1d, ema21_1d)
    
    # === 4h: Indicators ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 21 EMA for trend filter
    close_s = pd.Series(close)
    ema21 = close_s.ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # 100-period high/low for breakout (Donchian-like)
    high_max = pd.Series(high).rolling(window=100, min_periods=100).max().values
    low_min = pd.Series(low).rolling(window=100, min_periods=100).min().values
    
    # ATR(14) for volatility-based stop
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume spike detector (current volume > 2x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 100  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Get values
        ema21_1d_val = ema21_1d_aligned[i]
        ema21_val = ema21[i]
        high_max_val = high_max[i]
        low_min_val = low_min[i]
        atr_val = atr[i]
        vol_spike_val = vol_spike[i]
        current_close = close[i]
        current_high = high[i]
        current_low = low[i]
        
        # Skip if any value is NaN
        if (np.isnan(ema21_1d_val) or np.isnan(ema21_val) or 
            np.isnan(high_max_val) or np.isnan(low_min_val) or 
            np.isnan(atr_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above 100-period high with volume spike AND 1d EMA21 uptrend
            if (current_high > high_max_val and vol_spike_val and 
                ema21_1d_val > ema21_1d[max(0, i-1)] if i > 0 else ema21_1d_val > ema21_1d_val):
                signals[i] = 0.25
                position = 1
                entry_price = current_close
            
            # Short: break below 100-period low with volume spike AND 1d EMA21 downtrend
            elif (current_low < low_min_val and vol_spike_val and 
                  ema21_1d_val < ema21_1d[max(0, i-1)] if i > 0 else ema21_1d_val < ema21_1d_val):
                signals[i] = -0.25
                position = -1
                entry_price = current_close
        
        elif position == 1:
            # Long exit: break below 100-period low OR stop loss
            if (current_low < low_min_val or 
                current_close < entry_price - 2.5 * atr_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: break above 100-period high OR stop loss
            if (current_high > high_max_val or 
                current_close > entry_price + 2.5 * atr_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals