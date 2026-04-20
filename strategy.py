#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_RSI_Trend_200"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # === 1d: RSI(14) ===
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # === 4h: Indicators ===
    close = prices['close'].values
    volume = prices['volume'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # EMA200 for trend filter
    close_s = pd.Series(close)
    ema200 = close_s.ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # ATR(14) for stop loss
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 200  # Need enough data for EMA200
    
    for i in range(start_idx, n):
        # Get aligned values
        rsi = rsi_1d_aligned[i]
        current_ema200 = ema200[i]
        current_atr = atr[i]
        current_close = close[i]
        current_volume = volume[i]
        
        # Skip if any value is NaN
        if (np.isnan(rsi) or np.isnan(current_ema200) or np.isnan(current_atr)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume condition: current volume > 1.5x 20-period 4h average volume
        if i >= 20:
            vol_ma = np.mean(volume[i-20:i])
            vol_condition = current_volume > 1.5 * vol_ma
        else:
            vol_condition = False
        
        if position == 0:
            # Long: RSI < 30 (oversold) + price above EMA200 + volume
            if rsi < 30 and current_close > current_ema200 and vol_condition:
                signals[i] = 0.25
                position = 1
                entry_price = current_close
            # Short: RSI > 70 (overbought) + price below EMA200 + volume
            elif rsi > 70 and current_close < current_ema200 and vol_condition:
                signals[i] = -0.25
                position = -1
                entry_price = current_close
        
        elif position == 1:
            # Long exit: RSI > 70 (overbought) OR stop loss
            if rsi > 70 or current_close < entry_price - 2.0 * current_atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI < 30 (oversold) OR stop loss
            if rsi < 30 or current_close > entry_price + 2.0 * current_atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals