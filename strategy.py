#!/usr/bin/env python3
# 1d_1w_cci_volatility_filter_v1
# Strategy: 1d CCI with volatility filter on 1d and trend filter from 1w EMA
# Timeframe: 1d
# Leverage: 1.0
# Hypothesis: CCI identifies overbought/oversold conditions. Combined with 1d volatility filter
# (ATR ratio) to avoid low-volatility chop and 1w EMA trend filter to align with higher timeframe
# trend. This reduces false signals and improves win rate. Designed for low trade frequency
# (<25/year) to minimize fee drag in ranging markets like 2025.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_cci_volatility_filter_v1"
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
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1d CCI calculation (20-period)
    typical_price = (high + low + close) / 3
    tp_mean = pd.Series(typical_price).rolling(window=20, min_periods=20).mean().values
    tp_std = pd.Series(typical_price).rolling(window=20, min_periods=20).std().values
    # Avoid division by zero
    cci = np.where(tp_std != 0, (typical_price - tp_mean) / (0.015 * tp_std), 0.0)
    
    # 1d ATR for volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First TR is just high-low
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if np.isnan(cci[i]) or np.isnan(atr[i]) or np.isnan(ema_50_1w_aligned[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volatility filter: avoid low volatility (range) markets
        # ATR ratio: current ATR vs 50-period average ATR
        if i >= 50:
            atr_ma = pd.Series(atr[:i+1]).rolling(window=50, min_periods=50).mean().iloc[-1]
            vol_filter = atr[i] > 0.8 * atr_ma  # Only trade when volatility is above 80% of average
        else:
            vol_filter = True
        
        # CCI conditions
        cci_overbought = cci[i] > 100
        cci_oversold = cci[i] < -100
        cci_neutral = abs(cci[i]) <= 100
        
        # Trend filter: close vs 1w EMA50
        uptrend = close[i] > ema_50_1w_aligned[i]
        downtrend = close[i] < ema_50_1w_aligned[i]
        
        # Entry conditions
        # Long: CCI crosses below -100 (oversold) AND uptrend AND volatility filter
        if cci[i] < -100 and uptrend and vol_filter and position != 1:
            # Additional check: ensure we just crossed below -100 (was above or equal previous bar)
            if i == 20 or cci[i-1] >= -100:
                position = 1
                signals[i] = 0.25
        # Short: CCI crosses above 100 (overbought) AND downtrend AND volatility filter
        elif cci[i] > 100 and downtrend and vol_filter and position != -1:
            # Additional check: ensure we just crossed above 100 (was below or equal previous bar)
            if i == 20 or cci[i-1] <= 100:
                position = -1
                signals[i] = -0.25
        # Exit: CCI returns to neutral zone (|CCI| <= 100)
        elif position == 1 and cci_neutral:
            position = 0
            signals[i] = 0.0
        elif position == -1 and cci_neutral:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals