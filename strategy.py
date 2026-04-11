#!/usr/bin/env python3
# 1d_1w_keltner_channel_reversal_v1
# Strategy: Daily Keltner Channel reversal with weekly trend filter and volume confirmation
# Timeframe: 1d
# Leverage: 1.0
# Hypothesis: Keltner Channel reversions capture mean-reversion moves in extended trends; weekly EMA filter ensures alignment with higher-timeframe trend; volume confirmation avoids false signals. Designed for low trade frequency (<30/year) to minimize fee drag in BTC/ETH.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_keltner_channel_reversal_v1"
timeframe = "1d"
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
    
    # Load weekly data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly EMA(20) for trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Daily Keltner Channel (20, 2.0)
    typical_price = (high + low + close) / 3
    atr = pd.Series(high - low).rolling(window=20, min_periods=20).mean().values
    ema_tp = pd.Series(typical_price).ewm(span=20, adjust=False, min_periods=20).mean().values
    kc_upper = ema_tp + (2.0 * atr)
    kc_lower = ema_tp - (2.0 * atr)
    
    # Daily Volume confirmation: current volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_avg_20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(kc_upper[i]) or np.isnan(kc_lower[i]) or 
            np.isnan(ema_20_1w_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Keltner Channel reversals
        touch_upper = close[i] >= kc_upper[i]
        touch_lower = close[i] <= kc_lower[i]
        
        # Trend filter: price above/below weekly EMA20
        uptrend = close[i] > ema_20_1w_aligned[i]
        downtrend = close[i] < ema_20_1w_aligned[i]
        
        # Entry logic: Keltner Channel touch + volume + counter-trend
        if touch_lower and vol_confirm[i] and uptrend and position != 1:
            position = 1
            signals[i] = 0.25
        elif touch_upper and vol_confirm[i] and downtrend and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: opposite Keltner Channel touch with volume confirmation
        elif position == 1 and touch_upper and vol_confirm[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and touch_lower and vol_confirm[i]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals