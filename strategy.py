#!/usr/bin/env python3
"""
6h_keltner_rsi_divergence_1d_trend_v1
Hypothesis: Keltner Channel breakouts with RSI divergence on 6h, filtered by 1d EMA trend, work in both bull and bear markets.
- In uptrend (price > 1d EMA50): Look for bullish RSI divergence at lower Keltner band for longs
- In downtrend (price < 1d EMA50): Look for bearish RSI divergence at upper Keltner band for shorts
- Uses volatility-adjusted channels (ATR-based) which adapt to changing market conditions
- RSI divergence captures exhaustion moves before reversals
- Trend filter ensures we trade with the higher timeframe momentum
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_keltner_rsi_divergence_1d_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_price = prices['open'].values
    
    # 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Keltner Channel (20, 2.0) on 6h
    atr_period = 20
    ema_period = 20
    multiplier = 2.0
    
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # ATR
    atr = pd.Series(tr).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    
    # EMA of close
    ema_close = pd.Series(close).ewm(span=ema_period, adjust=False, min_periods=ema_period).mean().values
    
    # Keltner Bands
    upper_keltner = ema_close + (multiplier * atr)
    lower_keltner = ema_close - (multiplier * atr)
    
    # RSI (14) for divergence
    rsi_period = 14
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/rsi_period, adjust=False, min_periods=rsi_period).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/rsi_period, adjust=False, min_periods=rsi_period).mean().values
    
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, 50.0), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation: volume > 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if data not available
        if (np.isnan(upper_keltner[i]) or np.isnan(lower_keltner[i]) or 
            np.isnan(rsi[i]) or np.isnan(ema_50_aligned[i]) or
            np.isnan(close[i]) or np.isnan(volume[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            signals[i] = 0.0
            continue
        
        vol_confirmed = volume[i] > vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: RSI > 70 (overbought) or price crosses below EMA
            if rsi[i] > 70 or close[i] < ema_50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: RSI < 30 (oversold) or price crosses above EMA
            if rsi[i] < 30 or close[i] > ema_50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Need at least 3 periods for divergence check
            if i < 3:
                signals[i] = 0.0
                continue
                
            # Bullish RSI divergence: price makes lower low, RSI makes higher low
            bull_div = (low[i] < low[i-2] and rsi[i] > rsi[i-2])
            # Bearish RSI divergence: price makes higher high, RSI makes lower high
            bear_div = (high[i] > high[i-2] and rsi[i] < rsi[i-2])
            
            # Long: bullish divergence at lower Keltner band in uptrend
            if (close[i] > ema_50_aligned[i] and  # Uptrend filter
                low[i] <= lower_keltner[i] and   # Price at/touching lower band
                bull_div and                     # Bullish RSI divergence
                vol_confirmed):                  # Volume confirmation
                position = 1
                signals[i] = 0.25
            # Short: bearish divergence at upper Keltner band in downtrend
            elif (close[i] < ema_50_aligned[i] and  # Downtrend filter
                  high[i] >= upper_keltner[i] and   # Price at/touching upper band
                  bear_div and                      # Bearish RSI divergence
                  vol_confirmed):                   # Volume confirmation
                position = -1
                signals[i] = -0.25
    
    return signals