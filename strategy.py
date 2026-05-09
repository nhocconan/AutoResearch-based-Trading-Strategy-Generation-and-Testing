#!/usr/bin/env python3
# Hypothesis: 12h timeframe with 1-week Bollinger Band width percentile regime filter and 1-day RSI mean reversion.
# Uses weekly Bollinger Band width percentile to detect low-volatility squeeze (breakout favorable).
# Enters long when price breaks above upper Bollinger Band(20,2) on 1-day AND RSI(14) < 30 (oversold).
# Enters short when price breaks below lower Bollinger Band(20,2) on 1-day AND RSI(14) > 70 (overbought).
# Exits when price reverts to middle Bollinger Band(20) or volatility regime shifts to high volatility.
# Weekly volatility regime filter prevents whipsaws in high-volatility markets.
# Target: 50-150 total trades over 4 years (12-37/year) with size 0.25.

name = "12h_BBW_Percentile_RSI_MeanRev"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1-week Bollinger Band width percentile regime
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Bollinger Bands (20, 2) on weekly close
    bb_mid = pd.Series(df_1w['close']).rolling(window=20, min_periods=20).mean().values
    bb_std = pd.Series(df_1w['close']).rolling(window=20, min_periods=20).std().values
    bb_width = (bb_std * 2 * 2) / bb_mid  # (upper - lower) / middle = 4*std / middle
    
    # Percentile rank of BB width over 50-week lookback
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=50, min_periods=10).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else np.nan, raw=False
    ).values
    
    # Low volatility regime: BB width percentile < 30% (squeeze condition)
    vol_regime_squeeze = bb_width_percentile < 0.30
    vol_regime_squeeze_aligned = align_htf_to_ltf(prices, df_1w, vol_regime_squeeze)
    
    # Calculate 1-day RSI(14) for mean reversion signals
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # RSI(14) calculation
    delta = pd.Series(df_1d['close']).diff().values
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # RSI aligned to 12h timeframe
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Calculate 1-day Bollinger Bands (20, 2) for entry/exit
    bb_mid_1d = pd.Series(df_1d['close']).rolling(window=20, min_periods=20).mean().values
    bb_std_1d = pd.Series(df_1d['close']).rolling(window=20, min_periods=20).std().values
    bb_upper_1d = bb_mid_1d + (2 * bb_std_1d)
    bb_lower_1d = bb_mid_1d - (2 * bb_std_1d)
    
    # Align Bollinger Bands to 12h timeframe
    bb_mid_1d_aligned = align_htf_to_ltf(prices, df_1d, bb_mid_1d)
    bb_upper_1d_aligned = align_htf_to_ltf(prices, df_1d, bb_upper_1d)
    bb_lower_1d_aligned = align_htf_to_ltf(prices, df_1d, bb_lower_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_regime_squeeze_aligned[i]) or
            np.isnan(rsi_aligned[i]) or
            np.isnan(bb_mid_1d_aligned[i]) or np.isnan(bb_upper_1d_aligned[i]) or np.isnan(bb_lower_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: volatility squeeze + price above upper BB + RSI oversold
            if (vol_regime_squeeze_aligned[i] and
                close[i] > bb_upper_1d_aligned[i] and
                rsi_aligned[i] < 30):
                signals[i] = 0.25
                position = 1
            # Enter short: volatility squeeze + price below lower BB + RSI overbought
            elif (vol_regime_squeeze_aligned[i] and
                  close[i] < bb_lower_1d_aligned[i] and
                  rsi_aligned[i] > 70):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price reverts to middle BB OR volatility regime exits squeeze
            if (close[i] <= bb_mid_1d_aligned[i]) or (not vol_regime_squeeze_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price reverts to middle BB OR volatility regime exits squeeze
            if (close[i] >= bb_mid_1d_aligned[i]) or (not vol_regime_squeeze_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals