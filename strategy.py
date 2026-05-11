#!/usr/bin/env python3
"""
1D_1W_Trend_Follow_with_Regime_Filter_v1
Hypothesis: 1-day trend following with 1-week trend filter and volatility regime (ATR-based chop) filter.
Trades only when both timeframes agree on direction AND market is not choppy (ATR ratio < threshold).
Uses 20-period EMA for trend direction on both 1d and 1w. ATR ratio (current ATR / 20-period ATR avg) to filter chop.
Targets 10-25 trades per year (40-100 over 4 years) to minimize fee drag.
Works in bull/bear via trend following; avoids whipsaws via regime filter.
"""

name = "1D_1W_Trend_Follow_with_Regime_Filter_v1"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d and 1w data (HTF)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 30 or len(df_1w) < 10:
        return np.zeros(n)
    
    # 1d OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # --- 1d EMA(20) for trend ---
    ema_1d = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean()
    ema_1d_val = ema_1d.values
    
    # --- 1w EMA(20) for trend filter ---
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean()
    ema_1w_val = ema_1w.values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w_val)
    
    # --- ATR-based regime filter (chop detection) ---
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean()
    atr_val = atr.values
    
    # 20-period average ATR for regime
    atr_ma = pd.Series(atr_val).rolling(window=20, min_periods=20).mean()
    atr_ma_val = atr_ma.values
    # ATR ratio: current ATR / average ATR (low = chop, high = trending)
    atr_ratio = atr_val / (atr_ma_val + 1e-10)
    
    # Regime: trending when ATR ratio > 0.8 (avoid chop when ratio too low)
    trending_regime = atr_ratio > 0.8
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if 1w EMA not available (alignment)
        if np.isnan(ema_1w_aligned[i]):
            if position != 0:
                # Exit on trend break
                if position == 1 and close[i] < ema_1d_val[i]:
                    signals[i] = 0.0
                    position = 0
                elif position == -1 and close[i] > ema_1d_val[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25 if position == 1 else -0.25
            continue
        
        # 1d trend direction
        bullish_1d = close[i] > ema_1d_val[i]
        bearish_1d = close[i] < ema_1d_val[i]
        
        # 1w trend filter (must align with 1d)
        bullish_1w = ema_1w_aligned[i] > ema_1d_val[i]  # 1w EMA above 1d price = bullish
        bearish_1w = ema_1w_aligned[i] < ema_1d_val[i]  # 1w EMA below 1d price = bearish
        
        # Entry: both timeframes agree AND trending regime
        if position == 0:
            if bullish_1d and bullish_1w and trending_regime[i]:
                signals[i] = 0.25
                position = 1
            elif bearish_1d and bearish_1w and trending_regime[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: trend break on 1d OR regime turns choppy
            if position == 1:
                if (not bullish_1d) or (not trending_regime[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                if (not bearish_1d) or (not trending_regime[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals