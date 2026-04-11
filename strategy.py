#!/usr/bin/env python3
"""
1h_4d_vwap_reversion_v1
Strategy: 1h VWAP mean reversion with 4h trend filter and daily volatility regime
Timeframe: 1h
Leverage: 1.0
Hypothesis: In ranging markets (identified by low daily volatility), price reverts to VWAP.
In trending markets (identified by 4h EMA alignment), we follow the trend.
This dual approach works in both bull and bear markets by adapting to regime.
Uses 4h EMA for trend direction and daily ATR percentile for volatility regime.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4d_vwap_reversion_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate VWAP (typical price * volume) / cumulative volume
    typical_price = (high + low + close) / 3.0
    vwap = np.cumsum(typical_price * volume) / np.cumsum(volume)
    # Handle division by zero at start
    vwap = np.where(np.cumsum(volume) == 0, typical_price, vwap)
    
    # Load 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # 4h EMA(20) for trend direction
    ema_4h = pd.Series(df_4h['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Load daily data for volatility regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Daily ATR(14) for volatility measurement
    tr1 = df_1d['high'].values[1:] - df_1d['low'].values[1:]
    tr2 = np.abs(df_1d['high'].values[1:] - df_1d['close'].values[:-1])
    tr3 = np.abs(df_1d['low'].values[1:] - df_1d['close'].values[:-1])
    tr_1d = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14 = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Daily ATR percentile rank (20-day lookback) to identify regime
    atr_rank = pd.Series(atr_14).rolling(window=20, min_periods=10).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5, raw=False
    ).values
    atr_rank_aligned = align_htf_to_ltf(prices, df_1d, atr_rank)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(vwap[i]) or np.isnan(ema_4h_aligned[i]) or 
            np.isnan(atr_rank_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            # Outside session: flatten or hold flat
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        # Regime classification
        # Low volatility (ATR rank < 0.4) = range -> mean revert to VWAP
        # High volatility (ATR rank > 0.6) = trend -> follow 4h EMA
        atr_r = atr_rank_aligned[i]
        is_low_vol = atr_r < 0.4
        is_high_vol = atr_r > 0.6
        
        # Distance from VWAP (normalized by price)
        vwap_dist = (close[i] - vwap[i]) / vwap[i]
        
        # Entry logic based on regime
        if is_low_vol:
            # Mean reversion regime: fade extreme VWAP deviations
            if vwap_dist < -0.008 and position != 1:  # 0.8% below VWAP -> long
                position = 1
                signals[i] = 0.20
            elif vwap_dist > 0.008 and position != -1:  # 0.8% above VWAP -> short
                position = -1
                signals[i] = -0.20
            # Exit when price returns to VWAP (within 0.2%)
            elif abs(vwap_dist) < 0.002 and position != 0:
                position = 0
                signals[i] = 0.0
            else:
                # Hold position
                signals[i] = 0.20 if position == 1 else (-0.20 if position == -1 else 0.0)
                
        elif is_high_vol:
            # Trend following regime: follow 4h EMA
            if close[i] > ema_4h_aligned[i] and position != 1:
                position = 1
                signals[i] = 0.20
            elif close[i] < ema_4h_aligned[i] and position != -1:
                position = -1
                signals[i] = -0.20
            else:
                # Hold position
                signals[i] = 0.20 if position == 1 else (-0.20 if position == -1 else 0.0)
                
        else:
            # Neutral regime: stay flat to avoid whipsaw
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
    
    return signals