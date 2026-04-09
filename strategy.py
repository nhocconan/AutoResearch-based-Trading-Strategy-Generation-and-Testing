# BTC/ETH/SOL 4h strategy using 12h RSI divergence with 1d ATR-based position sizing
# Works in bull/bear via momentum divergence + volatility scaling
# Target: 20-40 trades/year per symbol with strong risk-adjusted returns

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_rsi_divergence_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE for RSI calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 14:
        return np.zeros(n)
    
    # Calculate 12h RSI(14) with proper Wilder's smoothing
    close_12h = df_12h['close'].values
    delta = np.diff(close_12h, prepend=close_12h[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    # Wilder's smoothing: first average is simple, then smoothed
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[13] = np.mean(gain[1:14])  # First 14-period average
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi_12h = 100 - (100 / (1 + rs))
    rsi_12h[:13] = np.nan  # Not enough data for first 13 periods
    
    # Align 12h RSI to 4h timeframe
    rsi_12h_aligned = align_htf_to_ltf(prices, df_12h, rsi_12h)
    
    # Load daily data for ATR-based position sizing
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate 14-day ATR
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr3 = np.abs(low_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_1d = np.zeros_like(tr)
    atr_1d[13] = np.mean(tr[1:14])
    for i in range(14, len(tr)):
        atr_1d[i] = (atr_1d[i-1] * 13 + tr[i]) / 14
    atr_1d[:13] = np.nan
    
    # Align daily ATR to 4h timeframe
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Price change over last 3 periods (12 hours) for momentum
    price_change = np.zeros(n)
    for i in range(3, n):
        price_change[i] = (close[i] - close[i-3]) / close[i-3]
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(rsi_12h_aligned[i]) or 
            np.isnan(rsi_12h_aligned[i-3]) or 
            np.isnan(price_change[i]) or 
            np.isnan(atr_1d_aligned[i]) or 
            atr_1d_aligned[i] <= 0):
            signals[i] = 0.0
            continue
        
        # RSI divergence signals
        rsi_now = rsi_12h_aligned[i]
        rsi_prev = rsi_12h_aligned[i-3]
        price_now = close[i]
        price_prev = close[i-3]
        
        # Bullish divergence: price makes lower low, RSI makes higher low
        bull_div = (price_now < price_prev) and (rsi_now > rsi_prev)
        # Bearish divergence: price makes higher high, RSI makes lower high
        bear_div = (price_now > price_prev) and (rsi_now < rsi_prev)
        
        if position == 0:  # Flat - look for new entries
            # Scale position size by volatility (inverse ATR)
            vol_factor = np.clip(1.0 / (atr_1d_aligned[i] * 0.01), 0.5, 2.0)
            base_size = 0.25
            size = base_size * vol_factor
            
            if bull_div and rsi_now < 40:  # Oversold condition for long
                position = 1
                signals[i] = size
            elif bear_div and rsi_now > 60:  # Overbought condition for short
                position = -1
                signals[i] = -size
                
        elif position == 1:  # Long position - exit on bearish divergence or RSI overbought
            if bear_div or rsi_now > 70:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain position
                
        elif position == -1:  # Short position - exit on bullish divergence or RSI oversold
            if bull_div or rsi_now < 30:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain position
    
    return signals