#!/usr/bin/env python3
# 12h_1w_keltner_breakout_volume_v1
# Strategy: 12h Keltner channel breakout with 1w trend filter and volume confirmation
# Timeframe: 12h
# Leverage: 1.0
# Hypothesis: Keltner channels (ATR-based) capture volatility expansion. Breakouts above upper band or below lower band
# with volume confirmation and 1w trend alignment capture high-probability moves. Works in bull markets via long breakouts
# and bear markets via short breakdowns. Designed for low trade frequency (~15-30/year) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1w_keltner_breakout_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # 12h ATR for Keltner channels
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # 12h EMA20 for middle band
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Keltner bands: upper = EMA + 2*ATR, lower = EMA - 2*ATR
    keltner_upper = ema_20 + 2.0 * atr
    keltner_lower = ema_20 - 2.0 * atr
    
    # 1w EMA40 for trend filter
    ema_40_1w = pd.Series(df_1w['close'].values).ewm(span=40, adjust=False, min_periods=40).mean().values
    ema_40_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_40_1w)
    
    # 12h volume average (20-period) for confirmation
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(40, n):
        # Skip if any required data is invalid
        if np.isnan(keltner_upper[i]) or np.isnan(keltner_lower[i]) or np.isnan(ema_40_1w_aligned[i]) or np.isnan(vol_avg_20[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume[i] > 1.5 * vol_avg_20[i]
        
        # Breakout signals
        breakout_up = high[i] > keltner_upper[i-1]
        breakdown_down = low[i] < keltner_lower[i-1]
        
        # 1w EMA trend filter: price above EMA = bullish trend, below = bearish
        trend_bullish = close[i] > ema_40_1w_aligned[i]
        trend_bearish = close[i] < ema_40_1w_aligned[i]
        
        # Entry conditions
        # Long: Breakout above upper band AND bullish trend AND volume confirmation
        if breakout_up and trend_bullish and vol_confirm and position != 1:
            position = 1
            signals[i] = 0.25
        # Short: Breakdown below lower band AND bearish trend AND volume confirmation
        elif breakdown_down and trend_bearish and vol_confirm and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: Opposite breakout (breakdown for long, breakout for short)
        elif position == 1 and breakdown_down:
            position = 0
            signals[i] = 0.0
        elif position == -1 and breakout_up:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals