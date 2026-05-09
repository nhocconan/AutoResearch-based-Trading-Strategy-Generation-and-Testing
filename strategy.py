#!/usr/bin/env python3
# Hypothesis: 12h timeframe with 1-day RSI(14) momentum and 1-week Bollinger Bands(20,2) regime.
# Enters long when RSI > 55 and price touches lower Bollinger Band in low volatility regime.
# Enters short when RSI < 45 and price touches upper Bollinger Band in low volatility regime.
# Uses 1-day ATR(14) normalized by 50-period mean to detect low-volatility regime (mean-reversion favorable).
# Weekly Bollinger Bands provide dynamic support/resistance that adapts to market conditions.
# Target: 50-150 total trades over 4 years (12-37/year) with size 0.25.

name = "12h_RSI_BBands_VolRegime"
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
    
    # Calculate 1-day RSI(14) for momentum
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # RSI(14)
    delta = np.diff(df_1d['close'], prepend=np.nan)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_14 = 100 - (100 / (1 + rs))
    
    # Calculate 1-day ATR(14) for volatility regime
    prev_close = np.roll(df_1d['close'], 1)
    prev_close[0] = np.nan
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - prev_close)
    tr3 = np.abs(df_1d['low'] - prev_close)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # 50-period mean of ATR for normalization
    atr_mean_50 = pd.Series(atr_14).rolling(window=50, min_periods=50).mean().values
    
    # Volatility regime: low volatility when ATR < 0.8 * mean ATR
    vol_regime_low = atr_14 < (0.8 * atr_mean_50)
    
    # Calculate 1-week Bollinger Bands(20,2) on 12h data (1 week = 14 bars)
    typical_price = (high + low + close) / 3.0
    bb_middle = pd.Series(typical_price).rolling(window=14, min_periods=14).mean().values
    bb_std = pd.Series(typical_price).rolling(window=14, min_periods=14).std().values
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    
    # Price touching Bollinger Bands (with small tolerance)
    price_touching_lower = low <= bb_lower * 1.001
    price_touching_upper = high >= bb_upper * 0.999
    
    # Align HTF indicators to 12h timeframe
    rsi_14_aligned = align_htf_to_ltf(prices, df_1d, rsi_14)
    vol_regime_low_aligned = align_htf_to_ltf(prices, df_1d, vol_regime_low)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(rsi_14_aligned[i]) or np.isnan(vol_regime_low_aligned[i]) or
            np.isnan(price_touching_lower[i]) or np.isnan(price_touching_upper[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: low volatility regime + RSI > 55 + price touching lower BB
            if vol_regime_low_aligned[i] and rsi_14_aligned[i] > 55 and price_touching_lower[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: low volatility regime + RSI < 45 + price touching upper BB
            elif vol_regime_low_aligned[i] and rsi_14_aligned[i] < 45 and price_touching_upper[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: volatility regime shifts to high OR RSI < 50 OR price touches upper BB
            if (not vol_regime_low_aligned[i]) or (rsi_14_aligned[i] < 50) or price_touching_upper[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: volatility regime shifts to high OR RSI > 50 OR price touches lower BB
            if (not vol_regime_low_aligned[i]) or (rsi_14_aligned[i] > 50) or price_touching_lower[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals