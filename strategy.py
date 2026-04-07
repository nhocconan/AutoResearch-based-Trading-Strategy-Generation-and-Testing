#!/usr/bin/env python3
"""
1h Momentum Reversal with 4h Trend Filter and Volume Spike
Hypothesis: Intraday reversals against short-term momentum, filtered by 4h trend,
             offer high-probability entries in both bull and bear markets.
             Volume spikes confirm institutional participation at turning points.
             Using 4h trend (not 1d) avoids whipsaws in choppy 1d trends.
             Target: 15-35 trades/year per symbol to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_momentum_reversal_4h_trend_volume_v1"
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
    
    # RSI(14) for momentum
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # 4h EMA20 Trend Filter
    df_4h = get_htf_data(prices, '4h')
    ema_20 = pd.Series(df_4h['close'].values).ewm(span=20, adjust=False).mean().values
    ema_20_aligned = align_htf_to_ltf(prices, df_4h, ema_20)
    
    # Volume Spike Detector (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    # Session filter: 08-20 UTC (already datetime64[ms] index)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(ema_20_aligned[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: RSI crosses above 60 (overbought) or trend turns bearish
            if rsi[i] > 60 or close[i] < ema_20_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: RSI crosses below 40 (oversold) or trend turns bullish
            if rsi[i] < 40 or close[i] > ema_20_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat, look for entry
            # Long: RSI < 30 (oversold) + price above 4h EMA20 + volume spike
            if (rsi[i] < 30 and 
                close[i] > ema_20_aligned[i] and 
                vol_spike[i]):
                position = 1
                signals[i] = 0.20
            # Short: RSI > 70 (overbought) + price below 4h EMA20 + volume spike
            elif (rsi[i] > 70 and 
                  close[i] < ema_20_aligned[i] and 
                  vol_spike[i]):
                position = -1
                signals[i] = -0.20
    
    return signals