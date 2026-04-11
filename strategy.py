#!/usr/bin/env python3
# 6h_1d_elder_ray_regime_v1
# Strategy: 6-period Elder Ray (Bull/Bear Power) with 1d trend filter and volatility regime filter
# Timeframe: 6h
# Leverage: 1.0
# Hypothesis: Elder Ray captures bull/bear power via EMA(13). Combined with 1d EMA50 trend filter 
# and ATR-based volatility regime (only trade in moderate volatility) to avoid whipsaws in low/high vol.
# Works in bull markets via bull power > 0 and in bear markets via bear power < 0.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_elder_ray_regime_v1"
timeframe = "6h"
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
    
    # 6-period EMA13 for Elder Ray calculation
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema13
    bear_power = low - ema13
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # ATR(20) for volatility regime filter
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    atr20 = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    # Volatility regime: only trade when ATR is between 20th and 80th percentile of last 100 periods
    # Avoid extremely low volatility (chop) and extremely high volatility (panic)
    vol_regime = np.ones(n, dtype=bool)
    for i in range(20, n):
        if i >= 100:
            atr_slice = atr20[i-100:i]
            if len(atr_slice) > 0:
                q20 = np.percentile(atr_slice, 20)
                q80 = np.percentile(atr_slice, 80)
                vol_regime[i] = (atr20[i] >= q20) and (atr20[i] <= q80)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if np.isnan(ema13[i]) or np.isnan(ema_50_1d_aligned[i]) or np.isnan(atr20[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Skip if not in volatility regime
        if not vol_regime[i]:
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Elder Ray signals
        bull_signal = bull_power[i] > 0  # Bull power positive
        bear_signal = bear_power[i] < 0  # Bear power negative
        
        # 1d EMA trend filter
        trend_bullish = close[i] > ema_50_1d_aligned[i]
        trend_bearish = close[i] < ema_50_1d_aligned[i]
        
        # Entry conditions
        # Long: Bull power > 0 AND bullish trend
        if bull_signal and trend_bullish and position != 1:
            position = 1
            signals[i] = 0.25
        # Short: Bear power < 0 AND bearish trend
        elif bear_signal and trend_bearish and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: Opposite Elder Ray signal
        elif position == 1 and not bull_signal:
            position = 0
            signals[i] = 0.0
        elif position == -1 and not bear_signal:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals