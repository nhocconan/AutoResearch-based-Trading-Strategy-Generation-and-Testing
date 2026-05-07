#!/usr/bin/env python3
"""
4h_Keltner_Channel_Breakout_12hTrend_Volume
Hypothesis: Keltner Channel breakout with 12-hour trend filter and volume confirmation.
In bull markets (price > 12h EMA50), long on upper band breakout with volume.
In bear markets (price < 12h EMA50), short on lower band breakout with volume.
Uses ATR-based bands for volatility adaptation, reducing false breakouts in low volatility.
Target: 25-40 trades per year (~100-160 over 4 years) with position size 0.25.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Keltner_Channel_Breakout_12hTrend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12-hour data ONCE for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # 12-hour EMA50 for trend filter
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Keltner Channel: 20-period EMA ± (2 * ATR(10))
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # True Range calculation
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0  # First period has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    upper_band = ema_20 + (2 * atr)
    lower_band = ema_20 - (2 * atr)
    
    # Volume ratio: current volume / 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.where(vol_ma > 0, volume / vol_ma, 1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need 50 periods for EMA50 and sufficient warmup
    
    for i in range(start_idx, n):
        if np.isnan(ema_50_12h_aligned[i]) or np.isnan(ema_20[i]) or np.isnan(atr[i]) or np.isnan(vol_ratio[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine market regime from 12-hour EMA50
        uptrend_regime = close[i] > ema_50_12h_aligned[i]
        downtrend_regime = close[i] < ema_50_12h_aligned[i]
        
        # Volume confirmation: volume > 1.5x average
        volume_confirm = vol_ratio[i] > 1.5
        
        if position == 0:
            # Long: close breaks above upper Keltner band in uptrend regime + volume
            long_entry = (close[i] > upper_band[i]) and uptrend_regime and volume_confirm
            # Short: close breaks below lower Keltner band in downtrend regime + volume
            short_entry = (close[i] < lower_band[i]) and downtrend_regime and volume_confirm
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: close crosses below EMA20 (middle band) or regime changes to downtrend
            if (close[i] < ema_20[i]) or (not uptrend_regime):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: close crosses above EMA20 (middle band) or regime changes to uptrend
            if (close[i] > ema_20[i]) or (not downtrend_regime):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals