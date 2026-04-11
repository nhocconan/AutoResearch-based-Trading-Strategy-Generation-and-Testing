#!/usr/bin/env python3
# 4h_12h_trix_volume_regime_v1
# Strategy: 4h TRIX momentum with 12h trend filter and volume confirmation
# Timeframe: 4h
# Leverage: 1.0
# Hypothesis: TRIX (TRIple Exponential Average) filters noise and identifies smooth momentum.
# Long when TRIX > 0 and rising, short when TRIX < 0 and falling, with 12h EMA trend filter.
# Volume spike (1.5x 20-period average) confirms momentum strength. Designed for moderate
# frequency (25-40 trades/year) to balance signal quality and fee drag in bull/bear markets.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_trix_volume_regime_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # 12h EMA(50) for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # TRIX calculation: Triple EMA of price, then ROC
    # TRIX = EMA(EMA(EMA(close, period), period), period) ROC
    period = 15
    ema1 = pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean()
    ema2 = ema1.ewm(span=period, adjust=False, min_periods=period).mean()
    ema3 = ema2.ewm(span=period, adjust=False, min_periods=period).mean()
    trix = ema3.pct_change(periods=1) * 100  # Rate of change
    
    # Volume spike: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_spike = volume > (vol_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(trix.iloc[i]) or 
            np.isnan(volume_spike[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend filter: price above/below 12h EMA50
        uptrend = close[i] > ema_50_12h_aligned[i]
        downtrend = close[i] < ema_50_12h_aligned[i]
        
        # TRIX signals: positive and rising = bullish momentum
        trix_now = trix.iloc[i]
        trix_prev = trix.iloc[i-1]
        trix_bullish = trix_now > 0 and trix_now > trix_prev
        trix_bearish = trix_now < 0 and trix_now < trix_prev
        
        # Entry logic: TRIX momentum + volume spike + trend alignment
        if (trix_bullish and volume_spike[i] and uptrend and position != 1):
            position = 1
            signals[i] = 0.25
        elif (trix_bearish and volume_spike[i] and downtrend and position != -1):
            position = -1
            signals[i] = -0.25
        # Exit: TRIX momentum reversal or trend change
        elif position == 1 and (not trix_bullish or not uptrend):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (not trix_bearish or not downtrend):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals