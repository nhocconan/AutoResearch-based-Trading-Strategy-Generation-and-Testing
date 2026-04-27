#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Use 1d RSI(14) extreme levels (oversold/overbought) as contrarian signals.
# Trade only during 08-20 UTC session to avoid low liquidity periods.
# Use 4h ATR for volatility filter to avoid choppy markets.
# Position size 0.20 to manage drawdown. Target: 15-30 trades/year.
# Works in bull/bear: mean reversion at extremes works in all regimes when combined with volatility filter.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for RSI calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d RSI(14)
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    # Prepend first value as NaN since diff reduces length by 1
    rsi_1d = np.concatenate([[np.nan], rsi_1d])
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Get 4h data for ATR volatility filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 14:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate 4h ATR(14)
    tr_4h = np.maximum(high_4h[1:] - low_4h[1:], 
                       np.maximum(np.abs(high_4h[1:] - close_4h[:-1]), 
                                  np.abs(low_4h[1:] - close_4h[:-1])))
    tr_4h = np.concatenate([[np.nan], tr_4h])
    atr_4h = pd.Series(tr_4h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    atr_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_4h)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if not session_mask[i]:
            signals[i] = 0.0
            continue
            
        if (np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(atr_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # RSI extremes: oversold < 30, overbought > 70
        rsi_oversold = rsi_1d_aligned[i] < 30
        rsi_overbought = rsi_1d_aligned[i] > 70
        
        # Volatility filter: only trade when ATR is above its 50-period median
        atr_median = np.nanmedian(atr_4h_aligned[max(0, i-50):i+1])
        vol_filter = atr_4h_aligned[i] > atr_median
        
        # Long when RSI oversold + volatility
        if rsi_oversold and vol_filter:
            signals[i] = 0.20
            position = 1
        # Short when RSI overbought + volatility
        elif rsi_overbought and vol_filter:
            signals[i] = -0.20
            position = -1
        # Exit when RSI returns to neutral range (40-60)
        elif position == 1 and rsi_1d_aligned[i] > 40:
            signals[i] = 0.0
            position = 0
        elif position == -1 and rsi_1d_aligned[i] < 60:
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_RSI14_Extreme_SessionVolFilter"
timeframe = "1h"
leverage = 1.0