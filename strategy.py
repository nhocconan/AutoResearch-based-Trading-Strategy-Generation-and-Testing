#!/usr/bin/env python3
# 1d_1w_keltner_breakout_v1
# Strategy: 1d Keltner Channel breakout with weekly trend filter and volume confirmation
# Timeframe: 1d
# Leverage: 1.0
# Hypothesis: Keltner breakouts capture volatility expansions. Weekly EMA200 filters trend direction.
# Volume > 1.5x 20-day average confirms institutional participation. Designed for low trade frequency
# (~10-25/year) to minimize fee drag. Works in bull markets via breakout continuation and bear markets
# via mean reversion when price breaks below lower band in downtrend.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_keltner_breakout_v1"
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
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 200:
        return np.zeros(n)
    
    # 1d ATR(10) for Keltner Channel
    tr1 = high - low
    tr2 = np.abs(np.concatenate([[high[0]], high[:-1]]) - np.concatenate([[close[0]], close[:-1]]))
    tr3 = np.abs(np.concatenate([[low[0]], low[:-1]]) - np.concatenate([[close[0]], close[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(alpha=1/10, adjust=False, min_periods=10).mean().values
    
    # 1d EMA(20) for Keltner Channel middle
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Keltner Channel bands
    upper_band = ema_20 + 2.0 * atr
    lower_band = ema_20 - 2.0 * atr
    
    # 20-day average volume for confirmation
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Weekly EMA200 for trend filter
    weekly_close = df_1w['close'].values
    ema_200_1w = pd.Series(weekly_close).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if np.isnan(ema_20[i]) or np.isnan(atr[i]) or np.isnan(vol_avg_20[i]) or np.isnan(ema_200_1w_aligned[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current volume > 1.5x 20-day average
        vol_confirm = volume[i] > 1.5 * vol_avg_20[i]
        
        # Weekly trend filter: price above/below weekly EMA200
        weekly_uptrend = close[i] > ema_200_1w_aligned[i]
        weekly_downtrend = close[i] < ema_200_1w_aligned[i]
        
        # Entry conditions
        # Long: price breaks above upper band AND weekly uptrend AND volume confirmation
        if close[i] > upper_band[i] and weekly_uptrend and vol_confirm and position != 1:
            position = 1
            signals[i] = 0.25
        # Short: price breaks below lower band AND weekly downtrend AND volume confirmation
        elif close[i] < lower_band[i] and weekly_downtrend and vol_confirm and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: price returns to middle EMA (mean reversion)
        elif position == 1 and close[i] < ema_20[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > ema_20[i]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals