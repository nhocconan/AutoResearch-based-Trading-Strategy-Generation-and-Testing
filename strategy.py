#!/usr/bin/env python3
# 1d_1w_cci_reversal_volume_v1
# Strategy: 1d CCI reversal with volume confirmation and 1w EMA trend filter
# Timeframe: 1d
# Leverage: 1.0
# Hypothesis: CCI > 100 indicates overbought conditions in uptrend, CCI < -100 indicates oversold in downtrend.
# In bull markets, look for CCI < -100 with volume spike and weekly uptrend for long entries.
# In bear markets, look for CCI > 100 with volume spike and weekly downtrend for short entries.
# Volume confirms reversal conviction. Weekly EMA filter ensures alignment with higher timeframe trend.
# Target: 20-40 trades/year to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_cci_reversal_volume_v1"
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
    
    # Load 1w data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1w EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Daily CCI(20)
    typical_price = (high + low + close) / 3.0
    tp_mean = pd.Series(typical_price).rolling(window=20, min_periods=20).mean()
    tp_dev = pd.Series(typical_price).rolling(window=20, min_periods=20).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=False
    )
    cci = (typical_price - tp_mean.values) / (0.015 * tp_dev.values)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_series = pd.Series(volume)
    vol_avg_20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (2.0 * vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(cci[i]) or np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend filter: price above/below 1w EMA50
        uptrend = close[i] > ema_50_1w_aligned[i]
        downtrend = close[i] < ema_50_1w_aligned[i]
        
        # Entry logic: CCI reversal + volume + trend alignment
        if cci[i] < -100 and vol_confirm[i] and uptrend and position != 1:
            position = 1
            signals[i] = 0.25
        elif cci[i] > 100 and vol_confirm[i] and downtrend and position != -1:
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