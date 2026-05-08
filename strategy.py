#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_Adaptive_VWAP_Momentum_1dTrend"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # 1d VWAP calculation (typical price * volume)
    tp_1d = (high_1d + low_1d + close_1d) / 3.0
    vwap_num = np.cumsum(tp_1d * volume_1d)
    vwap_den = np.cumsum(volume_1d)
    vwap_1d = np.divide(vwap_num, vwap_den, out=np.full_like(vwap_num, np.nan), where=vwap_den!=0)
    
    # 1d VWAP deviation (normalized by ATR-like volatility)
    atr_1d = pd.Series(high_1d - low_1d).rolling(window=14, min_periods=14).mean().values
    vwap_dev_1d = (close_1d - vwap_1d) / (atr_1d + 1e-10)
    
    # 1d trend: EMA50
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d indicators to 6h (wait for daily close)
    vwap_dev_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_dev_1d)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 6h momentum: RSI(9) for short-term momentum
    rsi_period = 9
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/rsi_period, adjust=False, min_periods=rsi_period).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/rsi_period, adjust=False, min_periods=rsi_period).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # 6h volume filter: volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(vwap_dev_1d_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(rsi[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above VWAP (bullish bias) + uptrend + bullish RSI + volume
            long_cond = (vwap_dev_1d_aligned[i] > 0.1) and \
                        (close[i] > ema_50_1d_aligned[i]) and \
                        (rsi[i] > 55) and \
                        volume_filter[i]
            # Short: price below VWAP (bearish bias) + downtrend + bearish RSI + volume
            short_cond = (vwap_dev_1d_aligned[i] < -0.1) and \
                         (close[i] < ema_50_1d_aligned[i]) and \
                         (rsi[i] < 45) and \
                         volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below VWAP or RSI overbought
            if (vwap_dev_1d_aligned[i] < -0.05) or (rsi[i] > 75):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above VWAP or RSI oversold
            if (vwap_dev_1d_aligned[i] > 0.05) or (rsi[i] < 25):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals