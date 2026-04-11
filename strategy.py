#!/usr/bin/env python3
# 4h_1d_keltner_channel_volume_v1
# Strategy: 4h Keltner Channel breakout with 1d volume confirmation and trend filter
# Timeframe: 4h
# Leverage: 1.0
# Hypothesis: Keltner Channel breakouts capture momentum with volatility-based bands. Volume confirmation ensures breakout strength. 
# Trend filter (1d EMA50) ensures alignment with higher timeframe trend. Designed for low trade frequency to minimize fee drift.
# Works in bull by riding uptrend breakouts, in bear by shorting downtrend breakdowns.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_keltner_channel_volume_v1"
timeframe = "4h"
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
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Keltner Channel on 4h (20-period, ATR multiplier 2.0)
    period = 20
    ma = pd.Series(close).rolling(window=period, min_periods=period).mean()
    atr = pd.Series(high - low).rolling(window=period, min_periods=period).mean()
    upper_band = (ma + 2.0 * atr).values
    lower_band = (ma - 2.0 * atr).values
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 1d volume average (20-period) for confirmation
    volume_1d = df_1d['volume'].values
    vol_avg_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    
    # Align raw 1d volume for current bar comparison
    vol_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or \
           np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_avg_20_1d_aligned[i]) or \
           np.isnan(vol_1d_aligned[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current 1d volume > 20-period average
        vol_confirm = vol_1d_aligned[i] > vol_avg_20_1d_aligned[i]
        
        # Trend filter: close vs 1d EMA50
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Entry conditions
        # Long: Price breaks above upper band AND uptrend AND volume confirmation
        if not np.isnan(upper_band[i]) and close[i] > upper_band[i] and uptrend and vol_confirm and position != 1:
            position = 1
            signals[i] = 0.25
        # Short: Price breaks below lower band AND downtrend AND volume confirmation
        elif not np.isnan(lower_band[i]) and close[i] < lower_band[i] and downtrend and vol_confirm and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: Price crosses opposite band
        elif position == 1 and close[i] < lower_band[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > upper_band[i]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals