#!/usr/bin/env python3
# 6h_1d_cci_trend_volume_v2
# Strategy: 6-hour CCI with 1-day trend filter and volume confirmation
# Timeframe: 6h
# Leverage: 1.0
# Hypothesis: CCI identifies overbought/oversold conditions. Combined with 1-day trend (EMA50) and volume spikes,
# it captures mean reversion in ranging markets and trend continuation in strong moves.
# Works in bull markets via trend-following longs and bear markets via trend-following shorts.
# Designed for low trade frequency (~15-30/year) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_cci_trend_volume_v2"
timeframe = "6h"
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
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 6h CCI(20)
    tp = (high + low + close) / 3.0
    sma_tp = pd.Series(tp).rolling(window=20, min_periods=20).mean().values
    mad = pd.Series(tp).rolling(window=20, min_periods=20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=False).values
    cci = (tp - sma_tp) / (0.015 * mad + 1e-10)
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 1d volume average (20-period) for confirmation
    volume_1d = df_1d['volume'].values
    vol_avg_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    
    # Align raw 1d volume for confirmation
    vol_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if np.isnan(cci[i]) or np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_avg_20_1d_aligned[i]) or np.isnan(vol_1d_aligned[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current 1d volume > 1.5x 20-period average
        vol_confirm = vol_1d_aligned[i] > 1.5 * vol_avg_20_1d_aligned[i]
        
        # Trend filter: price above/below 1d EMA50
        price_above_trend = close[i] > ema_50_1d_aligned[i]
        price_below_trend = close[i] < ema_50_1d_aligned[i]
        
        # CCI signals: oversold (< -100) or overbought (> 100)
        cci_oversold = cci[i] < -100
        cci_overbought = cci[i] > 100
        
        # Entry conditions
        # Long: CCI oversold AND price above 1d EMA50 (uptrend) AND volume confirmation
        if cci_oversold and price_above_trend and vol_confirm and position != 1:
            position = 1
            signals[i] = 0.25
        # Short: CCI overbought AND price below 1d EMA50 (downtrend) AND volume confirmation
        elif cci_overbought and price_below_trend and vol_confirm and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: CCI returns to neutral zone (-50 to 50)
        elif position == 1 and cci[i] > -50:
            position = 0
            signals[i] = 0.0
        elif position == -1 and cci[i] < 50:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals