#!/usr/bin/env python3
name = "4h_Donchian20_Breakout_12hTrend_Volume_Regime"
timeframe = "4h"
leverage = 1.0

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
    
    # Load 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # 12h EMA50 for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_avg)
    
    # ATR for volatility regime filter
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[0.0], np.maximum(np.maximum(tr1, tr2), tr3)])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_pct = atr / close
    # Volatility regime: not too low (<0.5%) and not too high (>6%)
    vol_regime = (atr_pct > 0.005) & (atr_pct < 0.06)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # ensure indicators have enough data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(vol_filter[i]) or np.isnan(vol_regime[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: breakout above Donchian high + above 12h EMA50 + volume filter + vol regime
            if close[i] > donchian_high[i] and close[i] > ema_50_12h_aligned[i] and vol_filter[i] and vol_regime[i]:
                signals[i] = 0.25
                position = 1
            # Short: breakdown below Donchian low + below 12h EMA50 + volume filter + vol regime
            elif close[i] < donchian_low[i] and close[i] < ema_50_12h_aligned[i] and vol_filter[i] and vol_regime[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: close below Donchian low or below 12h EMA50
            if close[i] < donchian_low[i] or close[i] < ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: close above Donchian high or above 12h EMA50
            if close[i] > donchian_high[i] or close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals