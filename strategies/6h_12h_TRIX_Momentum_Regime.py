#!/usr/bin/env python3
"""
6h_12h_TRIX_Momentum_Regime
Hypothesis: TRIX (triple-smoothed EMA) on 6h with 12h trend filter and volume confirmation.
- Long when: TRIX > 0, 12h EMA50 uptrend, volume > 20-period average
- Short when: TRIX < 0, 12h EMA50 downtrend, volume > 20-period average
- Exit when TRIX crosses zero or trend reverses
TRIX captures momentum with less lag than MACD. Works in bull by riding uptrends,
in bear by catching downtrends. Volume filter ensures momentum has participation.
Targets 15-35 trades/year (60-140 over 4 years) to minimize fee drag.
"""

name = "6h_12h_TRIX_Momentum_Regime"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # 6h OHLCV
    close_6h = prices['close'].values
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    volume_6h = prices['volume'].values
    
    # --- 12h Trend Filter: EMA50 ---
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # --- TRIX on 6h: triple EMA of ROC ---
    # ROC: (close - close_periods_ago) / close_periods_ago * 100
    period = 15
    roc = np.full_like(close_6h, np.nan)
    for i in range(period, len(close_6h)):
        roc[i] = (close_6h[i] - close_6h[i - period]) / close_6h[i - period] * 100.0
    
    # Triple EMA of ROC
    ema1 = pd.Series(roc).ewm(span=period, adjust=False, min_periods=period).mean().values
    ema2 = pd.Series(ema1).ewm(span=period, adjust=False, min_periods=period).mean().values
    ema3 = pd.Series(ema2).ewm(span=period, adjust=False, min_periods=period).mean().values
    trix = ema3  # TRIX is the final smoothed EMA
    
    # --- Volume Confirmation: 6h volume > 20-period average ---
    vol_ma_20 = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 60  # for ROC + triple EMA
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(trix[i]) or np.isnan(ema50_12h_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine 12h trend
        trend_up = close_6h[i] > ema50_12h_aligned[i]
        trend_down = close_6h[i] < ema50_12h_aligned[i]
        
        # Volume confirmation
        vol_ok = volume_6h[i] > vol_ma_20[i]
        
        if position == 0:
            # Look for entries only in direction of 12h trend with volume
            if trix[i] > 0 and trend_up and vol_ok:
                # Long: positive TRIX + 12h uptrend + volume
                signals[i] = 0.25
                position = 1
            elif trix[i] < 0 and trend_down and vol_ok:
                # Short: negative TRIX + 12h downtrend + volume
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit long: TRIX turns negative OR trend turns down
                if trix[i] < 0 or not trend_up:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: TRIX turns positive OR trend turns up
                if trix[i] > 0 or not trend_down:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals