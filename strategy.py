#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Choppiness Index (CI) + 12h ADX regime filter + 12h EMA50 trend.
# CI < 38.2 = trending (trend follow), CI > 61.8 = ranging (mean revert).
# In trending regime (CI < 38.2): go long when price > 12h EMA50, short when price < 12h EMA50.
# In ranging regime (CI > 61.8): go long at 12h EMA50 - 0.5*ATR(12h), short at 12h EMA50 + 0.5*ATR(12h).
# Uses 12h EMA50 and ATR for multi-timeframe alignment.
# Designed to avoid whipsaws in ranging markets and capture trends when present.
# Target: 15-30 trades/year to minimize fee drag.
name = "6h_Choppiness_ADX_EMA50_Regime"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # ATR(14) for choppiness calculation
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Choppiness Index (CI) = 100 * log10(sum(ATR(14)) / (n * ATR)) / log10(n)
    # Using 14-period lookback for CI
    atr_sum = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    n_period = 14
    ci = 100 * np.log10(atr_sum / (n_period * atr)) / np.log10(n_period)
    
    # 12h EMA50 and ATR for trend and mean reversion levels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # EMA50
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # ATR(12h) for mean reversion bands
    tr1_12h = high_12h[1:] - low_12h[1:]
    tr2_12h = np.abs(high_12h[1:] - close_12h[:-1])
    tr3_12h = np.abs(low_12h[1:] - close_12h[:-1])
    tr_12h = np.concatenate([[np.max([high_12h[0] - low_12h[0], np.abs(high_12h[0] - close_12h[0]), np.abs(low_12h[0] - close_12h[0])])], np.maximum(tr1_12h, np.maximum(tr2_12h, tr3_12h))])
    atr_12h = pd.Series(tr_12h).rolling(window=14, min_periods=14).mean().values
    atr_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Sufficient warmup for EMA50
    
    for i in range(start_idx, n):
        if (np.isnan(ci[i]) or np.isnan(atr[i]) or np.isnan(ema50_12h_aligned[i]) or 
            np.isnan(atr_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Regime determination
        trending = ci[i] < 38.2
        ranging = ci[i] > 61.8
        
        if position == 0:
            if trending:
                # Trend following: long above EMA50, short below EMA50
                if close[i] > ema50_12h_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                elif close[i] < ema50_12h_aligned[i]:
                    signals[i] = -0.25
                    position = -1
            elif ranging:
                # Mean reversion: long at EMA50 - 0.5*ATR, short at EMA50 + 0.5*ATR
                lower_band = ema50_12h_aligned[i] - 0.5 * atr_12h_aligned[i]
                upper_band = ema50_12h_aligned[i] + 0.5 * atr_12h_aligned[i]
                if close[i] < lower_band:
                    signals[i] = 0.25
                    position = 1
                elif close[i] > upper_band:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: regime change to ranging OR price crosses above EMA50 in trend
            if ranging or (trending and close[i] > ema50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: regime change to ranging OR price crosses below EMA50 in trend
            if ranging or (trending and close[i] < ema50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals