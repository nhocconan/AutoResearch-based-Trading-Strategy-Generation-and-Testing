#!/usr/bin/env python3
# 12h_1d_keltner_channel_v1
# Strategy: 12h Keltner Channel breakout with 1d ATR-based trend filter and volume confirmation
# Timeframe: 12h
# Leverage: 1.0
# Hypothesis: Keltner Channel breakouts capture volatility expansion moves. Combined with 1d ATR trend filter (using ATR > its 50-period MA to confirm trending markets) and volume confirmation, it works in both bull and bear markets by entering only during volatile, trending conditions while avoiding chop. Uses discrete position sizing to minimize fee churn.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_keltner_channel_v1"
timeframe = "12h"
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
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d ATR(14) and its 50-period MA for trend filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0  # First period has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma_50 = pd.Series(atr_14).rolling(window=50, min_periods=50).mean().values
    # Trend filter: ATR > its MA indicates expanding volatility/trending market
    atr_trend = atr_14 > atr_ma_50
    atr_trend_aligned = align_htf_to_ltf(prices, df_1d, atr_trend)
    
    # 12h Keltner Channel (20-period EMA, 2*ATR)
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    # ATR for KC (using same period as EMA smoothing)
    atr_kc = pd.Series(close).rolling(window=20, min_periods=20).apply(
        lambda x: np.max(np.abs(np.diff(np.concatenate(([x[0]], x))))), raw=True
    ).values
    # Simpler ATR calculation for KC
    tr_kc = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr_kc[0] = 0
    atr_kc = pd.Series(tr_kc).rolling(window=20, min_periods=20).mean().values
    
    kc_upper = ema_20 + 2 * atr_kc
    kc_lower = ema_20 - 2 * atr_kc
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_ratio = pd.Series(volume) / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_20[i]) or np.isnan(kc_upper[i]) or np.isnan(kc_lower[i]) or 
            np.isnan(atr_trend_aligned[i]) or np.isnan(vol_ratio.iloc[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current volume > 1.5x average
        vol_confirmed = vol_ratio.iloc[i] > 1.5
        
        # Entry conditions
        # Long: Close breaks above KC upper + ATR trend (volatile market) + volume confirmation
        if vol_confirmed and close[i] > kc_upper[i] and atr_trend_aligned[i] and position != 1:
            position = 1
            signals[i] = 0.25
        # Short: Close breaks below KC lower + ATR trend (volatile market) + volume confirmation
        elif vol_confirmed and close[i] < kc_lower[i] and atr_trend_aligned[i] and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit conditions: loss of volatility expansion or mean reversion to middle
        elif position == 1 and (close[i] < ema_20[i] or not atr_trend_aligned[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > ema_20[i] or not atr_trend_aligned[i]):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals