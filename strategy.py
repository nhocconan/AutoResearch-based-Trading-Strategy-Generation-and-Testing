#!/usr/bin/env python3
# 4h_12h_keltner_breakout_volume_v1
# Strategy: 4h Keltner breakout with 12h trend filter and volume confirmation
# Timeframe: 4h
# Leverage: 1.0
# Hypothesis: Keltner Channel breakouts capture volatility expansions. Using 12h EMA trend filter ensures trades align with higher timeframe momentum, reducing whipsaws. Volume confirmation filters low-momentum breakouts. Designed for 25-40 trades/year to minimize fee drag while capturing strong trends in both bull and bear markets.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_keltner_breakout_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate Keltner Channel (20, 2.0) on 4h data
    # Middle = EMA(20) of close
    # Upper = Middle + 2.0 * ATR(20)
    # Lower = Middle - 2.0 * ATR(20)
    close_series = pd.Series(close)
    ema_middle = close_series.ewm(span=20, adjust=False, min_periods=20).values
    
    # Calculate ATR(20)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=20, adjust=False, min_periods=20).values
    
    keltner_upper = ema_middle + 2.0 * atr
    keltner_lower = ema_middle - 2.0 * atr
    
    # 12h EMA50 for trend filter
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # 20-period volume average for confirmation
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(keltner_upper[i]) or np.isnan(keltner_lower[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_avg_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume[i] > 1.5 * vol_avg_20[i]
        
        # Breakout signals using Keltner Channel
        # Breakout above upper band
        breakout_up = high[i] > keltner_upper[i-1]
        # Breakdown below lower band
        breakdown_down = low[i] < keltner_lower[i-1]
        
        # 12h EMA trend filter: price above EMA = bullish trend, below = bearish
        trend_bullish = close[i] > ema_50_12h_aligned[i]
        trend_bearish = close[i] < ema_50_12h_aligned[i]
        
        # Entry conditions
        # Long: Breakout above upper band AND bullish trend AND volume confirmation
        if breakout_up and trend_bullish and vol_confirm and position != 1:
            position = 1
            signals[i] = 0.25
        # Short: Breakdown below lower band AND bearish trend AND volume confirmation
        elif breakdown_down and trend_bearish and vol_confirm and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: Opposite breakout using the middle line (mean reversion)
        elif position == 1 and close[i] < ema_middle[i-1]:  # Close below EMA middle
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > ema_middle[i-1]:  # Close above EMA middle
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals