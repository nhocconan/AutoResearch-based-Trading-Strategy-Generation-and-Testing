#!/usr/bin/env python3
# 1d_1w_keltner_channel_v1
# Strategy: 1d Keltner Channel breakout with 1w EMA trend filter and volume confirmation
# Timeframe: 1d
# Leverage: 1.0
# Hypothesis: Keltner Channel captures volatility-based breakouts. Combined with 1w EMA trend filter and volume confirmation, it works in both bull and bear markets by identifying strong momentum with institutional confirmation while avoiding false signals in choppy markets.

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
    
    # 1w EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Keltner Channel (20-period ATR, 2.0 multiplier)
    atr_period = 20
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    
    ma = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    upper = ma + 2.0 * atr
    lower = ma - 2.0 * atr
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_ratio = pd.Series(volume) / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(upper[i]) or 
            np.isnan(lower[i]) or np.isnan(vol_ratio.iloc[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current volume > 1.5x average
        vol_confirmed = vol_ratio.iloc[i] > 1.5
        
        # Entry conditions
        # Long: Close above upper Keltner band + price above 1w EMA50 (uptrend) + volume confirmation
        if vol_confirmed and close[i] > upper[i] and close[i] > ema_50_1w_aligned[i] and position != 1:
            position = 1
            signals[i] = 0.25
        # Short: Close below lower Keltner band + price below 1w EMA50 (downtrend) + volume confirmation
        elif vol_confirmed and close[i] < lower[i] and close[i] < ema_50_1w_aligned[i] and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit conditions: loss of momentum or trend reversal
        elif position == 1 and (close[i] < ma[i] or close[i] < ema_50_1w_aligned[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > ma[i] or close[i] > ema_50_1w_aligned[i]):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals