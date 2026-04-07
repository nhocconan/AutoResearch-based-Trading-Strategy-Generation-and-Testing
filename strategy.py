#!/usr/bin/env python3
"""
1h RSI Pullback with 4h Trend and Volume Confirmation
Long: 4h EMA50 > EMA200 (uptrend) + RSI(14) < 30 (oversold) + volume > 1.5x average
Short: 4h EMA50 < EMA200 (downtrend) + RSI(14) > 70 (overbought) + volume > 1.5x average
Exit: RSI crosses back to 50 or trend reverses
Session filter: 08-20 UTC only
Position size: 0.20
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_rsi_pullback_4h_trend_volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === 1h RSI(14) ===
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # === 4h EMA Trend (50 and 200) ===
    df_4h = get_htf_data(prices, '4h')
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False).mean().values
    ema_200_4h = pd.Series(df_4h['close'].values).ewm(span=200, adjust=False).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    ema_200_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_200_4h)
    
    # === Volume filter: 1.5x 20-period average ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if outside session
        if not (8 <= hours[i] <= 20):
            signals[i] = 0.0
            continue
            
        # Skip if not enough data
        if np.isnan(rsi[i]) or np.isnan(ema_50_4h_aligned[i]) or np.isnan(ema_200_4h_aligned[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
            
        vol_ok = volume[i] > 1.5 * vol_ma[i]
        uptrend = ema_50_4h_aligned[i] > ema_200_4h_aligned[i]
        downtrend = ema_50_4h_aligned[i] < ema_200_4h_aligned[i]
        
        if position == 1:  # Long position
            # Exit: RSI crosses above 50 or trend turns down
            if rsi[i] > 50 and rsi[i-1] <= 50:
                position = 0
                signals[i] = 0.0
            elif not uptrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: RSI crosses below 50 or trend turns up
            if rsi[i] < 50 and rsi[i-1] >= 50:
                position = 0
                signals[i] = 0.0
            elif not downtrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat, look for entry
            if vol_ok:
                # Long: uptrend + RSI oversold
                if uptrend and rsi[i] < 30 and rsi[i-1] >= 30:
                    position = 1
                    signals[i] = 0.20
                # Short: downtrend + RSI overbought
                elif downtrend and rsi[i] > 70 and rsi[i-1] <= 70:
                    position = -1
                    signals[i] = -0.20
    
    return signals