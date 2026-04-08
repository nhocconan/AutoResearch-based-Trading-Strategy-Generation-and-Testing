#!/usr/bin/env python3
# 6h_volume_squeeze_1d_trend_v1
# Hypothesis: On 6h timeframe, trade volatility breakouts when 6h ATR contracts then expands,
# filtered by 1d trend (EMA50). Low volatility precedes explosive moves; breakout in trend direction.
# Works in bull/bear markets as volatility expansion precedes trend continuation.
# Target: 20-40 trades/year.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_volume_squeeze_1d_trend_v1"
timeframe = "6h"
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
    
    # 6h ATR(20) for volatility measurement
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    # 6h ATR ratio: current ATR / 20-period average ATR (volatility contraction/expansion)
    atr_ma = pd.Series(atr).rolling(window=20, min_periods=20).mean().values
    atr_ratio = atr / atr_ma  # < 0.8 = contraction, > 1.2 = expansion
    
    # 6h Donchian breakout levels (20-period)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 1d EMA50 trend filter
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 50:
        return np.zeros(n)
    
    daily_close = df_daily['close'].values
    daily_ema50 = pd.Series(daily_close).ewm(span=50, min_periods=50, adjust=False).mean().values
    daily_ema50_6h = align_htf_to_ltf(prices, df_daily, daily_ema50)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if data not available
        if np.isnan(atr_ratio[i]) or np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or \
           np.isnan(daily_ema50_6h[i]):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below 6h EMA20 or ATR expansion fades
            ema20 = pd.Series(close[:i+1]).ewm(span=20, min_periods=20, adjust=False).mean().iloc[-1]
            if close[i] < ema20 or atr_ratio[i] < 1.0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above 6h EMA20 or ATR expansion fades
            ema20 = pd.Series(close[:i+1]).ewm(span=20, min_periods=20, adjust=False).mean().iloc[-1]
            if close[i] > ema20 or atr_ratio[i] < 1.0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volatility expansion trigger: ATR ratio > 1.2 (expansion from contraction)
            vol_expansion = atr_ratio[i] > 1.2
            
            # Breakout confirmation: price breaks Donchian level
            breakout_up = close[i] > donch_high[i]
            breakout_down = close[i] < donch_low[i]
            
            # 1d trend filter
            daily_uptrend = close[i] > daily_ema50_6h[i]
            daily_downtrend = close[i] < daily_ema50_6h[i]
            
            # Long entry: volatility expansion + upward breakout + uptrend
            if vol_expansion and breakout_up and daily_uptrend:
                position = 1
                signals[i] = 0.25
            # Short entry: volatility expansion + downward breakout + downtrend
            elif vol_expansion and breakout_down and daily_downtrend:
                position = -1
                signals[i] = -0.25
    
    return signals