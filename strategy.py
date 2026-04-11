#!/usr/bin/env python3
# 1d_1w_keltner_channel_v1
# Strategy: 1d Keltner Channel breakout with 1w EMA trend filter and volume confirmation
# Timeframe: 1d
# Leverage: 1.0
# Hypothesis: Keltner Channel breakouts capture volatility expansion; 1w EMA filters direction; volume confirms strength.
# Designed for low frequency (7-25 trades/year) to minimize fee decay in trending and ranging markets.
# Works in bull (breakouts with trend) and bear (mean reversion at extremes with volume spike).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_keltner_channel_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # 1w EMA(20) for trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Keltner Channel (20, 2.0)
    atr_period = 20
    tr = np.maximum(high[1:] - low[1:], np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
    tr = np.insert(tr, 0, 0)
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean()
    
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean()
    
    upper_kc = ema20 + 2.0 * atr
    lower_kc = ema20 - 2.0 * atr
    
    # Volume confirmation: 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(ema_20_1w_aligned[i]) or np.isnan(upper_kc.iloc[i]) or 
            np.isnan(lower_kc.iloc[i]) or np.isnan(vol_ma20.iloc[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend filter: 1w EMA direction
        uptrend = close[i] > ema_20_1w_aligned[i]
        downtrend = close[i] < ema_20_1w_aligned[i]
        
        # Volume confirmation: above average
        vol_confirm = volume[i] > vol_ma20.iloc[i]
        
        # Entry conditions
        if uptrend and vol_confirm and close[i] > upper_kc.iloc[i-1] and position != 1:
            position = 1
            signals[i] = 0.25
        elif downtrend and vol_confirm and close[i] < lower_kc.iloc[i-1] and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit conditions: opposite breakout or trend reversal
        elif position == 1 and (close[i] < lower_kc.iloc[i-1] or not uptrend):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > upper_kc.iloc[i-1] or not downtrend):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals