#!/usr/bin/env python3
# 12h_1w_cci_trend_volume_v1
# Strategy: 12h CCI trend following with 1w EMA filter and volume confirmation
# Timeframe: 12h
# Leverage: 1.0
# Hypothesis: CCI identifies overbought/oversold conditions while trend follows higher timeframe EMA.
# Volume confirms institutional participation. Designed for low trade frequency (<30/year) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1w_cci_trend_volume_v1"
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
    
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 12h CCI calculation
    typical_price = (high + low + close) / 3
    tp_mean = pd.Series(typical_price).rolling(window=20, min_periods=20).mean().values
    tp_std = pd.Series(typical_price).rolling(window=20, min_periods=20).std().values
    # Avoid division by zero
    tp_std = np.where(tp_std == 0, 1e-10, tp_std)
    cci = (typical_price - tp_mean) / (0.015 * tp_std)
    
    # 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # 1w volume average (20-period) for confirmation
    volume_1w = df_1w['volume'].values
    vol_avg_20_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_avg_20_1w)
    
    # Align raw 1w volume for confirmation
    vol_1w_aligned = align_htf_to_ltf(prices, df_1w, volume_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if np.isnan(cci[i]) or np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_avg_20_1w_aligned[i]) or np.isnan(vol_1w_aligned[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current 1w volume > 1.3x 20-period average
        vol_confirm = vol_1w_aligned[i] > 1.3 * vol_avg_20_1w_aligned[i]
        
        # Trend filter: price vs 1w EMA50
        uptrend = close[i] > ema_50_1w_aligned[i]
        downtrend = close[i] < ema_50_1w_aligned[i]
        
        # CCI conditions
        cci_overbought = cci[i] > 100
        cci_oversold = cci[i] < -100
        cci_neutral = (cci[i] >= -100) & (cci[i] <= 100)
        
        # Entry conditions
        # Long: CCI crosses above -100 from oversold AND uptrend AND volume confirmation
        if cci[i] > -100 and (i == 50 or cci[i-1] <= -100) and uptrend and vol_confirm and position != 1:
            position = 1
            signals[i] = 0.25
        # Short: CCI crosses below 100 from overbought AND downtrend AND volume confirmation
        elif cci[i] < 100 and (i == 50 or cci[i-1] >= 100) and downtrend and vol_confirm and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: CCI returns to neutral zone (mean reversion)
        elif position == 1 and cci[i] < 0:
            position = 0
            signals[i] = 0.0
        elif position == -1 and cci[i] > 0:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals