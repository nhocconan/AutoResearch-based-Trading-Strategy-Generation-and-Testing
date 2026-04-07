#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 1h RSI Pullback with 4h Trend Filter and Session Filter
# Hypothesis: RSI pullbacks in direction of 4h EMA(20) trend capture mean reversion within trends.
# Uses 4h EMA for trend filter and RSI(14) for precise entries. Session filter (08-20 UTC) reduces noise.
# Target: 15-37 trades/year (60-150 total over 4 years) to minimize fee drag.

name = "1h_rsi_pullback_4h_trend_session_v1"
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
    
    # Pre-compute session hours (08-20 UTC) - assumes DatetimeIndex
    hours = prices.index.hour
    
    # Get 4h data for EMA trend
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate EMA(20) on 4h close
    close_4h = df_4h['close'].values
    ema_20 = pd.Series(close_4h).ewm(span=20, adjust=False).mean().values
    
    # Align 4h EMA to 1h
    ema_20_aligned = align_htf_to_ltf(prices, df_4h, ema_20)
    
    # RSI(14) on 1h close
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.where(avg_loss == 0, 100, rsi)  # up days only
    rsi = np.where(avg_gain == 0, 0, rsi)    # down days only
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if np.isnan(ema_20_aligned[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
            
        # Session filter: 08-20 UTC
        if not (8 <= hours[i] <= 20):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: RSI reaches overbought or trend changes
            if rsi[i] >= 70 or close[i] < ema_20_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20  # Maintain long
        elif position == -1:  # Short position
            # Exit: RSI reaches oversold or trend changes
            if rsi[i] <= 30 or close[i] > ema_20_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20  # Maintain short
        else:  # Flat, look for entry
            # RSI pullback in direction of 4h trend
            if close[i] > ema_20_aligned[i]:  # Uptrend
                if rsi[i] <= 30:  # Pullback to buy (oversold)
                    position = 1
                    signals[i] = 0.20
            else:  # Downtrend
                if rsi[i] >= 70:  # Pullback to sell (overbought)
                    position = -1
                    signals[i] = -0.20
    
    return signals