#!/usr/bin/env python3
name = "1d_WeeklyDonchian_Breakout_1wTrend"
timeframe = "1d"
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
    
    # Load weekly data for trend and Donchian channels
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    volume_1w = df_1w['volume'].values
    
    # Weekly Donchian channels (20-week lookback)
    donch_high_20 = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donch_low_20 = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Weekly EMA for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align weekly indicators to daily timeframe
    donch_high_20_aligned = align_htf_to_ltf(prices, df_1w, donch_high_20)
    donch_low_20_aligned = align_htf_to_ltf(prices, df_1w, donch_low_20)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Daily volume filter: current volume > 1.5x 20-day average
    vol_avg_20d = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_avg_20d)
    
    # ATR filter to avoid extreme volatility
    tr1 = np.maximum(high[1:] - low[1:], np.absolute(high[1:] - close[:-1]))
    tr2 = np.maximum(np.absolute(low[1:] - close[:-1]), tr1)
    tr = np.concatenate([[tr1[0]], tr2]) if len(tr1) > 0 else np.array([0.0])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_pct = atr / close
    vol_regime = (atr_pct > 0.01) & (atr_pct < 0.05)  # reasonable volatility range
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # ensure indicators have enough data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donch_high_20_aligned[i]) or 
            np.isnan(donch_low_20_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(vol_filter[i]) or np.isnan(vol_regime[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: breakout above weekly Donchian high + above weekly EMA50 + volume filter + vol regime
            if high[i] > donch_high_20_aligned[i] and close[i] > ema_50_1w_aligned[i] and vol_filter[i] and vol_regime[i]:
                signals[i] = 0.25
                position = 1
            # Short: breakdown below weekly Donchian low + below weekly EMA50 + volume filter + vol regime
            elif low[i] < donch_low_20_aligned[i] and close[i] < ema_50_1w_aligned[i] and vol_filter[i] and vol_regime[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: breakdown below weekly Donchian low or below weekly EMA50
            if low[i] < donch_low_20_aligned[i] or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: breakout above weekly Donchian high or above weekly EMA50
            if high[i] > donch_high_20_aligned[i] or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals