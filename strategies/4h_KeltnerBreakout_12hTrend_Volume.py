#!/usr/bin/env python3
"""
4h_KeltnerBreakout_12hTrend_Volume
Hypothesis: Keltner channel breakouts with 12h EMA50 trend filter and volume confirmation capture trends in both bull and bear markets.
Keltner channels (ATR-based) adapt to volatility, reducing false signals during high volatility periods. Targets 20-40 trades/year on 4h to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Calculate ATR for Keltner channels (20-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Keltner channels: EMA20 ± 2*ATR
    ema20 = pd.Series(close).ewm(span=20, adjust=False).mean().values
    upper_keltner = ema20 + 2 * atr
    lower_keltner = ema20 - 2 * atr
    
    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for EMA, ATR, and volume MA
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if np.isnan(ema50_12h_aligned[i]) or np.isnan(atr[i]):
            signals[i] = 0.0
            continue
        
        ema_trend = ema50_12h_aligned[i]
        vol_confirm_val = vol_confirm[i]
        
        if position == 0:
            # Long: break above upper Keltner with uptrend and volume
            if close[i] > upper_keltner[i] and vol_confirm_val and close[i] > ema_trend:
                signals[i] = size
                position = 1
            # Short: break below lower Keltner with downtrend and volume
            elif close[i] < lower_keltner[i] and vol_confirm_val and close[i] < ema_trend:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: break below lower Keltner or trend turns down
            if close[i] < lower_keltner[i] or close[i] < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: break above upper Keltner or trend turns up
            if close[i] > upper_keltner[i] or close[i] > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_KeltnerBreakout_12hTrend_Volume"
timeframe = "4h"
leverage = 1.0