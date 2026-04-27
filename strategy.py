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
    
    # Get weekly data for trend and volatility
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly EMA(20) for trend filter
    ema20_1w = pd.Series(df_1w['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Weekly ATR(14) for volatility
    tr1_w = df_1w['high'].values - df_1w['low'].values
    tr2_w = np.abs(df_1w['high'].values - np.roll(df_1w['close'].values, 1))
    tr3_w = np.abs(df_1w['low'].values - np.roll(df_1w['close'].values, 1))
    tr_w = np.maximum(tr1_w, np.maximum(tr2_w, tr3_w))
    tr_w[0] = tr1_w[0]
    atr_1w_raw = pd.Series(tr_w).rolling(window=14, min_periods=14).mean().values
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_1w_raw)
    
    # Daily ATR(14) for volatility filter
    tr1_d = high - low
    tr2_d = np.abs(high - np.roll(close, 1))
    tr3_d = np.abs(low - np.roll(close, 1))
    tr_d = np.maximum(tr1_d, np.maximum(tr2_d, tr3_d))
    tr_d[0] = tr1_d[0]
    atr_1d = pd.Series(tr_d).rolling(window=14, min_periods=14).mean().values
    
    # Daily volume ratio (current / 20-day average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / (vol_ma + 1e-10)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup
    start_idx = max(20, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema20_1w_aligned[i]) or np.isnan(atr_1w_aligned[i]) or 
            np.isnan(atr_1d[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        ema_trend = ema20_1w_aligned[i]
        atr_1w_val = atr_1w_aligned[i]
        atr_1d_val = atr_1d[i]
        vol_ratio_val = vol_ratio[i]
        
        # Volatility filter: daily ATR > 0.3 * weekly ATR (higher volatility regime)
        vol_filter = atr_1d_val > (atr_1w_val * 0.3)
        
        # Volume filter: current volume > 1.5 * 20-day average
        vol_confirm = vol_ratio_val > 1.5
        
        if position == 0:
            # Long: price above weekly EMA with volatility and volume confirmation
            if close[i] > ema_trend and vol_filter and vol_confirm:
                signals[i] = size
                position = 1
            # Short: price below weekly EMA with volatility and volume confirmation
            elif close[i] < ema_trend and vol_filter and vol_confirm:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below weekly EMA
            if close[i] < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above weekly EMA
            if close[i] > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_EMA20_Trend_VolumeFilter_v1"
timeframe = "1d"
leverage = 1.0