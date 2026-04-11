#!/usr/bin/env python3
# 1h_4h_1d_volume_momentum_v1
# Strategy: 1h momentum with 4h/1d volume and trend filters
# Timeframe: 1h
# Leverage: 1.0
# Hypothesis: Momentum works best when aligned with higher timeframe trend and volume confirmation.
# Uses 4h for trend direction (EMA50), 1d for volume regime (volume > 1.5x 20-day average),
# and 1h for entry timing (RSI pullback in trend direction). Designed for 15-37 trades/year.
# Works in bull markets via trend continuation and in bear markets via mean-reversion pullbacks
# within the dominant trend, avoiding chop via volume regime filter.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h_1d_volume_momentum_v1"
timeframe = "1h"
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
    
    # Session filter: 08-20 UTC (precomputed)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 4h data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # 4h EMA(50) for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1d volume regime: volume > 1.5x 20-day average
    vol_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_regime = pd.Series(vol_1d) / pd.Series(vol_ma_20)
    vol_regime_aligned = align_htf_to_ltf(prices, df_1d, vol_regime.values)
    
    # 1h RSI(14) for entry timing
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values  # Neutral when undefined
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Warmup for indicators
        # Skip if any required data is invalid or outside session
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(vol_regime_aligned[i]) or 
            np.isnan(rsi[i]) or not in_session[i]):
            # Flatten position if outside session or invalid data
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        # Volume regime filter: only trade when volume is elevated
        vol_filter = vol_regime_aligned[i] > 1.5
        
        # Entry conditions
        # Long: Uptrend (price > 4h EMA50) + RSI pullback (30 < RSI < 50) + volume regime
        if vol_filter and close[i] > ema_50_4h_aligned[i] and 30 < rsi[i] < 50 and position != 1:
            position = 1
            signals[i] = 0.20
        # Short: Downtrend (price < 4h EMA50) + RSI bounce (50 < RSI < 70) + volume regime
        elif vol_filter and close[i] < ema_50_4h_aligned[i] and 50 < rsi[i] < 70 and position != -1:
            position = -1
            signals[i] = -0.20
        # Exit conditions: trend reversal or RSI extreme
        elif position == 1 and (close[i] < ema_50_4h_aligned[i] or rsi[i] >= 70):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > ema_50_4h_aligned[i] or rsi[i] <= 30):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.20 if position == 1 else (-0.20 if position == -1 else 0.0)
    
    return signals