#!/usr/bin/env python3
"""
1h_RSI_Divergence_4hTrend_VolumeConfirm_v1
Hypothesis: On 1h timeframe, bullish/bearish RSI divergences signal reversals. 4h EMA50 provides trend filter for context alignment. Volume spike (>1.8x 20-period EMA) confirms entry validity. Session filter (08-20 UTC) reduces noise. Target 20-40 trades/year per symbol, using discrete position sizing (0.20) to control fee drag and drawdown. Works in bull/bear by trading reversals within the trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 4h data ONCE before loop for HTF trend filter (EMA)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema_50 = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50)
    
    # Calculate RSI(14) on 1h
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values  # neutral when undefined
    
    # Volume spike detection on 1h (volume > 1.8x 20-period EMA)
    volume_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (volume_ema * 1.8)
    
    # Session filter: 08-20 UTC (precomputed for efficiency)
    hours = prices.index.hour  # prices.index is DatetimeIndex
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need sufficient data for RSI and EMA)
    start_idx = max(50, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready or outside session
        if (np.isnan(ema_50_aligned[i]) or 
            np.isnan(rsi[i]) or
            not in_session[i]):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        # 4h trend filter (EMA50)
        uptrend = close[i] > ema_50_aligned[i]
        downtrend = close[i] < ema_50_aligned[i]
        
        # Bullish RSI divergence: price makes lower low, RSI makes higher low
        bullish_div = False
        if i >= 2:
            if low[i] < low[i-1] and low[i-1] < low[i-2] and rsi[i] > rsi[i-1] and rsi[i-1] > rsi[i-2]:
                bullish_div = True
        
        # Bearish RSI divergence: price makes higher high, RSI makes lower high
        bearish_div = False
        if i >= 2:
            if high[i] > high[i-1] and high[i-1] > high[i-2] and rsi[i] < rsi[i-1] and rsi[i-1] < rsi[i-2]:
                bearish_div = True
        
        # Long logic: bullish divergence + volume spike + in uptrend
        if bullish_div and volume_spike[i] and uptrend:
            if position != 1:
                signals[i] = 0.20
                position = 1
            else:
                signals[i] = 0.20
        # Short logic: bearish divergence + volume spike + in downtrend
        elif bearish_div and volume_spike[i] and downtrend:
            if position != -1:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = -0.20
        # Exit conditions: opposite divergence or trend weakening
        elif position == 1 and (bearish_div or not uptrend):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (bullish_div or not downtrend):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_RSI_Divergence_4hTrend_VolumeConfirm_v1"
timeframe = "1h"
leverage = 1.0