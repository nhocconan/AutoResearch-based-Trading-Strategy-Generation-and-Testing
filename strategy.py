#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
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
    
    # 12h ATR(14) for volatility filter
    tr1_h = high - low
    tr2_h = np.abs(high - np.roll(close, 1))
    tr3_h = np.abs(low - np.roll(close, 1))
    tr_h = np.maximum(tr1_h, np.maximum(tr2_h, tr3_h))
    tr_h[0] = tr1_h[0]
    atr_12h = pd.Series(tr_h).rolling(window=14, min_periods=14).mean().values
    
    # 12h RSI(14) for momentum confirmation
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup
    start_idx = max(20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema20_1w_aligned[i]) or np.isnan(atr_12h[i]) or 
            i >= len(atr_1w_aligned) or np.isnan(atr_1w_aligned[i]) or
            np.isnan(rsi[i])):
            signals[i] = 0.0
            continue
        
        ema_trend = ema20_1w_aligned[i]
        atr_12h_val = atr_12h[i]
        atr_1w_val = atr_1w_aligned[i]
        rsi_val = rsi[i]
        
        # Volatility filter: 12h ATR > 0.5 * weekly ATR (higher volatility regime)
        vol_filter = atr_12h_val > (atr_1w_val * 0.5)
        
        # RSI filter: avoid overbought/oversold extremes
        rsi_filter = (rsi_val > 30) & (rsi_val < 70)
        
        if position == 0:
            # Long: price above EMA with volatility and RSI filter
            if close[i] > ema_trend and vol_filter and rsi_filter:
                signals[i] = size
                position = 1
            # Short: price below EMA with volatility and RSI filter
            elif close[i] < ema_trend and vol_filter and rsi_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below EMA
            if close[i] < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above EMA
            if close[i] > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_EMA20_Trend_VolumeRSIFilter_v1"
timeframe = "12h"
leverage = 1.0