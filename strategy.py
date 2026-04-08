#!/usr/bin/env python3
# 4h_rsi_momentum_v1
# Hypothesis: RSI momentum strategy with volume and volatility filters. Uses RSI(14) to identify overbought/oversold conditions,
# enters on mean reversion during low volatility and momentum continuation during high volatility.
# Includes volume confirmation and ATR-based stop management. Designed for 4H timeframe to work in both bull and bear markets.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_rsi_momentum_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily trend filter (1d EMA50) - load once before loop
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA50 on daily data
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 4h indicators
    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # ATR(14) for volatility
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Volume confirmation
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 50  # Need indicators warmed up
    
    for i in range(start_idx, n):
        if np.isnan(rsi[i]) or np.isnan(atr[i]) or np.isnan(avg_volume[i]) or np.isnan(ema50_1d_aligned[i]):
            if position != 0:
                pass
            else:
                signals[i] = 0.0
            continue
        
        # Daily trend filter
        daily_uptrend = close[i] > ema50_1d_aligned[i]
        daily_downtrend = close[i] < ema50_1d_aligned[i]
        
        if position == 1:  # Long position
            # Exit conditions: RSI overbought or volatility contraction
            if rsi[i] > 70 or atr[i] < atr[i-1] * 0.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions: RSI oversold or volatility contraction
            if rsi[i] < 30 or atr[i] < atr[i-1] * 0.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume confirmation
            volume_ok = volume[i] > 1.2 * avg_volume[i]
            
            if volume_ok:
                # Mean reversion in low volatility
                if atr[i] < np.nanmedian(atr[max(0, i-50):i+1]):
                    # Long when oversold in uptrend
                    if daily_uptrend and rsi[i] < 30:
                        position = 1
                        signals[i] = 0.25
                    # Short when overbought in downtrend
                    elif daily_downtrend and rsi[i] > 70:
                        position = -1
                        signals[i] = -0.25
                # Momentum continuation in high volatility
                else:
                    # Long when bullish momentum in uptrend
                    if daily_uptrend and rsi[i] > 50 and rsi[i] > rsi[i-1]:
                        position = 1
                        signals[i] = 0.25
                    # Short when bearish momentum in downtrend
                    elif daily_downtrend and rsi[i] < 50 and rsi[i] < rsi[i-1]:
                        position = -1
                        signals[i] = -0.25
    
    return signals