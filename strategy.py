#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for indicators
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Daily RSI(14) for trend filter
    delta = pd.Series(df_1d['close']).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rs = rs.replace([np.inf, -np.inf], 100)
    rsi_14 = 100 - (100 / (1 + rs))
    rsi_14_vals = rsi_14.values
    rsi_14_aligned = align_htf_to_ltf(prices, df_1d, rsi_14_vals)
    
    # Daily ATR (14-period) for volatility filter
    tr1 = pd.Series(df_1d['high'] - df_1d['low'])
    tr2 = pd.Series(np.abs(df_1d['high'] - df_1d['close'].shift(1)))
    tr3 = pd.Series(np.abs(df_1d['low'] - df_1d['close'].shift(1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    tr.iloc[0] = 0
    atr_14 = tr.ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # 4h EMA(50) for trend filter
    ema50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume confirmation: volume > 1.3 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for EMA50, volume MA, and RSI
    start_idx = max(50, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if np.isnan(rsi_14_aligned[i]) or np.isnan(ema50[i]) or np.isnan(atr_14_aligned[i]):
            signals[i] = 0.0
            continue
        
        rsi_val = rsi_14_aligned[i]
        ema_val = ema50[i]
        vol_spike_val = vol_spike[i]
        atr_val = atr_14_aligned[i]
        
        if position == 0:
            # Long: price crosses above EMA50 + volume spike + bullish momentum (RSI > 55)
            if close[i] > ema_val and close[i-1] <= ema_val and vol_spike_val and rsi_val > 55:
                signals[i] = size
                position = 1
            # Short: price crosses below EMA50 + volume spike + bearish momentum (RSI < 45)
            elif close[i] < ema_val and close[i-1] >= ema_val and vol_spike_val and rsi_val < 45:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below EMA50 or momentum turns bearish
            if close[i] < ema_val or rsi_val < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above EMA50 or momentum turns bullish
            if close[i] > ema_val or rsi_val > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_EMA50_RSI_Volume_Filter_v1"
timeframe = "4h"
leverage = 1.0