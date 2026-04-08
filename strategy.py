#!/usr/bin/env python3
"""
1d_keltner_rsi_reversion_v1
Hypothesis: Mean reversion on daily timeframe using Keltner Channel + RSI.
- Uses 20-day EMA with ATR(10) bands (Keltner Channel)
- Long when price touches lower band and RSI < 30 (oversold)
- Short when price touches upper band and RSI > 70 (overbought)
- Exit when price crosses back to EMA or RSI normalizes
- Weekly trend filter: only take long if weekly close > weekly EMA(50), short if < weekly EMA(50)
- Designed for low trade frequency (<20/year) to minimize fee drag
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_keltner_rsi_reversion_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # === DAILY INDICATORS ===
    # EMA(20) for Keltner middle
    close_s = pd.Series(close)
    ema_20 = close_s.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # ATR(10) for Keltner width
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = 0  # first bar has no previous close
    atr_10 = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Keltner Bands
    keltner_upper = ema_20 + 2 * atr_10
    keltner_lower = ema_20 - 2 * atr_10
    
    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # === WEEKLY TREND FILTER ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    weekly_bullish = close_1w > ema_50_1w
    weekly_bearish = close_1w < ema_50_1w
    
    # Align weekly trend to daily
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish.astype(float))
    weekly_bearish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bearish.astype(float))
    
    # === SIGNAL LOGIC ===
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after EMA warmup
        # Skip if data not ready
        if (np.isnan(keltner_upper[i]) or np.isnan(keltner_lower[i]) or 
            np.isnan(rsi[i]) or np.isnan(weekly_bullish_aligned[i]) or 
            np.isnan(weekly_bearish_aligned[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long
            # Exit: price crosses above EMA(20) or RSI > 50
            if close[i] > ema_20[i] or rsi[i] > 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: price crosses below EMA(20) or RSI < 50
            if close[i] < ema_20[i] or rsi[i] < 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long: price touches lower band + RSI oversold + weekly bullish
            if (low[i] <= keltner_lower[i] and 
                rsi[i] < 30 and 
                weekly_bullish_aligned[i] > 0.5):
                position = 1
                signals[i] = 0.25
            # Short: price touches upper band + RSI overbought + weekly bearish
            elif (high[i] >= keltner_upper[i] and 
                  rsi[i] > 70 and 
                  weekly_bearish_aligned[i] > 0.5):
                position = -1
                signals[i] = -0.25
    
    return signals