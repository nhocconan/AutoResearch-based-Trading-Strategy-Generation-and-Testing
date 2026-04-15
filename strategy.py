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
    
    # Get 1d HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily ATR(14) for volatility filter and stoploss
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - np.concatenate([[df_1d['close'].iloc[0]], df_1d['close'].iloc[:-1]]))
    tr3 = np.abs(df_1d['low'] - np.concatenate([[df_1d['close'].iloc[0]], df_1d['close'].iloc[:-1]]))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate daily EMA(50) for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate daily RSI(14) for momentum filter
    delta = pd.Series(df_1d['close'].values).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_14_1d = 100 - (100 / (1 + rs))
    rsi_14_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_14_1d)
    
    # Track position for stoploss
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_14_1d_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(rsi_14_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        current_price = close[i]
        
        # Check stoploss for existing position
        if position == 1 and current_price < entry_price - 1.5 * atr_14_1d_aligned[i]:
            signals[i] = 0.0
            position = 0
            continue
        elif position == -1 and current_price > entry_price + 1.5 * atr_14_1d_aligned[i]:
            signals[i] = 0.0
            position = 0
            continue
        
        # Regime filter: only trade when daily ATR is elevated (> 0.3% of price)
        vol_filter = atr_14_1d_aligned[i] > 0.003 * current_price
        
        # Long conditions:
        # 1. Price above daily EMA50 (bullish bias)
        # 2. Daily RSI between 30 and 70 (avoid extremes)
        # 3. Volatility filter
        if (current_price > ema_50_1d_aligned[i] and
            30 <= rsi_14_1d_aligned[i] <= 70 and
            vol_filter and
            position != 1):
            signals[i] = 0.25
            position = 1
            entry_price = current_price
            
        # Short conditions:
        # 1. Price below daily EMA50 (bearish bias)
        # 2. Daily RSI between 30 and 70 (avoid extremes)
        # 3. Volatility filter
        elif (current_price < ema_50_1d_aligned[i] and
              30 <= rsi_14_1d_aligned[i] <= 70 and
              vol_filter and
              position != -1):
            signals[i] = -0.25
            position = -1
            entry_price = current_price
        else:
            signals[i] = 0.0
            # Keep current position
    
    return signals

name = "12h_EMA50_RSI14_VolFilter_SL_v1"
timeframe = "12h"
leverage = 1.0