#!/usr/bin/env python3
"""
1d Bollinger Squeeze with 1w Volume Confirmation
Hypothesis: Bollinger Bands squeeze indicates low volatility and impending breakout.
Volume confirmation filters breakouts. Works in bull markets (breakouts up) and bear markets (breakouts down).
Target: 20-40 trades/year (80-160 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "14358_1d_bollinger_squeeze_vol_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1w data for volume confirmation (once before loop)
    df_1w = get_htf_data(prices, '1w')
    vol_1w = df_1w['volume'].values
    
    # Daily data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Bollinger Bands (20, 2)
    bb_period = 20
    bb_std = 2
    sma = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean()
    std = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std()
    upper = sma + (std * bb_std)
    lower = sma - (std * bb_std)
    
    # Bollinger Band Width (normalized)
    bb_width = (upper - lower) / sma
    bb_width_s = pd.Series(bb_width)
    
    # Bollinger Band Width percentile (20-period lookback)
    bb_width_percentile = bb_width_s.rolling(window=20, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5, raw=False
    ).values
    
    # Volume filter: 1w volume above 50th percentile of last 4 weeks
    vol_1w_s = pd.Series(vol_1w)
    vol_percentile = vol_1w_s.rolling(window=4, min_periods=4).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5, raw=False
    ).values
    vol_filter_1w = vol_percentile > 0.5  # Above median volume
    
    # Align 1w indicators to daily timeframe
    bb_width_percentile_aligned = align_htf_to_ltf(prices, df_1w, bb_width_percentile)
    vol_filter_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_filter_1w.astype(float))
    
    # ATR for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(bb_period, 20) + 4  # BB period + volume lookback
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(bb_width_percentile_aligned[i]) or 
            np.isnan(vol_filter_1w_aligned[i]) or 
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits
        if position == 1:  # long position
            # Exit: price below middle band OR stoploss
            if (close[i] <= sma[i] or close[i] <= entry_price - 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price above middle band OR stoploss
            if (close[i] >= sma[i] or close[i] >= entry_price + 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Bollinger squeeze (low volatility) + volume breakout
            squeeze = bb_width_percentile_aligned[i] < 0.2  # Bottom 20% of BB width
            vol_ok = vol_filter_1w_aligned[i] > 0.5  # Above median 1w volume
            
            if squeeze and vol_ok:
                # Breakout direction based on price vs SMA
                if close[i] > sma[i]:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                elif close[i] < sma[i]:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0
    
    return signals