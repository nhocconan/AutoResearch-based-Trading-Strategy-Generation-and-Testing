#!/usr/bin/env python3
# 1d_weekly_ema_trend_v1
# Hypothesis: Daily trend following with weekly EMA filter and ATR-based position sizing.
# Uses 1d EMA50 for trend direction and 1w EMA20 for higher timeframe confirmation.
# Position size scaled by ATR volatility (inverse volatility targeting) to maintain consistent risk.
# Works in bull/bear: trend filter avoids counter-trend trades, ATR scaling reduces size in high volatility.
# Target: 15-30 trades/year (60-120 total over 4 years).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_weekly_ema_trend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 1d EMA50 for trend direction
    ema50_1d = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # 1w HTF data for higher timeframe trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:  # Need sufficient data for EMA20
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # ATR(14) for volatility-based position sizing
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema50_1d[i]) or np.isnan(ema20_1w_aligned[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: trend turns bearish (price below both EMAs)
            if close[i] < ema50_1d[i] or close[i] < ema20_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                # Scale position by inverse ATR (volatility targeting)
                atr_norm = atr[i] / np.nanmedian(atr[max(0, i-100):i+1])  # Normalize ATR
                size = 0.30 / (1 + atr_norm)  # Reduce size in high volatility
                signals[i] = np.clip(size, 0.10, 0.30)
                
        elif position == -1:  # Short position
            # Exit: trend turns bullish (price above both EMAs)
            if close[i] > ema50_1d[i] or close[i] > ema20_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                # Scale position by inverse ATR (volatility targeting)
                atr_norm = atr[i] / np.nanmedian(atr[max(0, i-100):i+1])  # Normalize ATR
                size = 0.30 / (1 + atr_norm)  # Reduce size in high volatility
                signals[i] = -np.clip(size, 0.10, 0.30)
        else:  # Flat
            # Enter long: price above both EMAs (bullish alignment)
            if close[i] > ema50_1d[i] and close[i] > ema20_1w_aligned[i]:
                position = 1
                # Scale position by inverse ATR (volatility targeting)
                atr_norm = atr[i] / np.nanmedian(atr[max(0, i-100):i+1])  # Normalize ATR
                size = 0.30 / (1 + atr_norm)  # Reduce size in high volatility
                signals[i] = np.clip(size, 0.10, 0.30)
            # Enter short: price below both EMAs (bearish alignment)
            elif close[i] < ema50_1d[i] and close[i] < ema20_1w_aligned[i]:
                position = -1
                # Scale position by inverse ATR (volatility targeting)
                atr_norm = atr[i] / np.nanmedian(atr[max(0, i-100):i+1])  # Normalize ATR
                size = 0.30 / (1 + atr_norm)  # Reduce size in high volatility
                signals[i] = -np.clip(size, 0.10, 0.30)
    
    return signals