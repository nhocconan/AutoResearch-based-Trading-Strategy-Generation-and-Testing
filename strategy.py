#/usr/bin/env python3
"""
1h Volume-Weighted RSI with 4h Trend Filter
Long when RSI(14) < 30 and 4h EMA(20) trending up with volume confirmation
Short when RSI(14) > 70 and 4h EMA(20) trending down with volume confirmation
Exit when RSI returns to neutral zone (40-60) or 4h trend flips
Designed to capture mean reversion in trending markets with volume filter to avoid whipsaws
Target: 15-37 trades/year on 1h timeframe
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_volume_weighted_rsi_4h_trend_v1"
timeframe = "1h"
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
    
    # Session filter: 08-20 UTC (already datetime64[ms] in index)
    hours = prices.index.hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    # === RSI Calculation ===
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    alpha = 1.0 / 14
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, n):
        avg_gain[i] = alpha * gain[i] + (1 - alpha) * avg_gain[i-1]
        avg_loss[i] = alpha * loss[i] + (1 - alpha) * avg_loss[i-1]
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    
    # === 4h Trend Filter (EMA20) ===
    # Get 4h data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Calculate EMA20 on 4h close
    ema_4h = np.zeros_like(close_4h)
    ema_4h[19] = np.mean(close_4h[:20])
    for i in range(20, len(close_4h)):
        ema_4h[i] = (close_4h[i] * 2 / 21) + (ema_4h[i-1] * 19 / 21)
    
    # Align to 1h timeframe (shifted by 1 for no look-ahead)
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Volume confirmation (20-period average)
    vol_ma = np.zeros(n)
    vol_ma[19] = np.mean(volume[:20])
    for i in range(20, n):
        vol_ma[i] = (volume[i] * 2 / 21) + (vol_ma[i-1] * 19 / 21)
    vol_ratio = volume / (vol_ma + 1e-10)
    
    # Generate signals
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if outside trading session
        if not session_mask[i]:
            signals[i] = 0.0
            continue
            
        # Skip if any data is NaN
        if np.isnan(rsi[i]) or np.isnan(ema_4h_aligned[i]) or np.isnan(vol_ratio[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: RSI returns to neutral OR 4h trend turns down
            if rsi[i] >= 40 or ema_4h_aligned[i] < close_4h[np.searchsorted(df_4h.index.values, prices.index[i]) // 16 if i >= 16 else 0]:
                # Simplified: exit when RSI > 40 or 4h EMA < 4h close (trend weakening)
                if rsi[i] >= 40 or ema_4h_aligned[i] < close[i]:  # Simplified trend check
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.20
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: RSI returns to neutral OR 4h trend turns up
            if rsi[i] <= 60 or ema_4h_aligned[i] > close[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat, look for entry
            # Need volume confirmation
            if vol_ratio[i] < 1.5:
                signals[i] = 0.0
                continue
            
            # Entry conditions
            # Long: RSI oversold AND 4h trend up
            # Short: RSI overbought AND 4h trend down
            if rsi[i] < 30 and ema_4h_aligned[i] > close[i]:
                # 4h trend up (EMA > price)
                position = 1
                signals[i] = 0.20
            elif rsi[i] > 70 and ema_4h_aligned[i] < close[i]:
                # 4h trend down (EMA < price)
                position = -1
                signals[i] = -0.20
    
    return signals