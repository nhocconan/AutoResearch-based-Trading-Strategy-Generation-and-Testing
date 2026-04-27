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
    
    # Get daily data for pivot levels and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily EMA(34) for trend filter
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate daily ATR(14) for volatility
    df_1d_high = df_1d['high'].values
    df_1d_low = df_1d['low'].values
    df_1d_close = df_1d['close'].values
    tr1 = df_1d_high - df_1d_low
    tr2 = np.abs(df_1d_high - np.roll(df_1d_close, 1))
    tr3 = np.abs(df_1d_low - np.roll(df_1d_close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate 4-hour ATR(14) for volatility filter
    tr1_h = high - low
    tr2_h = np.abs(high - np.roll(close, 1))
    tr3_h = np.abs(low - np.roll(close, 1))
    tr_h = np.maximum(tr1_h, np.maximum(tr2_h, tr3_h))
    tr_h[0] = tr1_h[0]
    atr_4h = pd.Series(tr_h).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 4-hour RSI(14)
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
    start_idx = max(34, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(atr_1d_aligned[i]) or 
            np.isnan(atr_4h[i]) or np.isnan(rsi[i])):
            signals[i] = 0.0
            continue
        
        ema_trend = ema34_1d_aligned[i]
        atr_1d_val = atr_1d_aligned[i]
        atr_4h_val = atr_4h[i]
        rsi_val = rsi[i]
        
        # Volatility filter: 4h ATR > daily ATR (higher volatility regime)
        vol_filter = atr_4h_val > (atr_1d_val * 0.5)
        
        # RSI filter: avoid extreme overbought/oversold
        rsi_filter = (rsi_val > 20) and (rsi_val < 80)
        
        if position == 0:
            # Long: price above EMA with volatility and RSI filters
            if (close[i] > ema_trend and vol_filter and rsi_filter):
                signals[i] = size
                position = 1
            # Short: price below EMA with volatility and RSI filters
            elif (close[i] < ema_trend and vol_filter and rsi_filter):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below EMA or RSI overbought
            if close[i] < ema_trend or rsi_val >= 75:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above EMA or RSI oversold
            if close[i] > ema_trend or rsi_val <= 25:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_EMA34_Trend_VolumeRSIFilter_v1"
timeframe = "4h"
leverage = 1.0