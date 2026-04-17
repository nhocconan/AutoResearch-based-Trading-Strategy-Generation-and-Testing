#!/usr/bin/env python3
"""
Hypothesis: 1h RSI(2) mean reversion with 4h trend filter and 1d volatility regime.
Long when RSI(2) < 10 AND price > 4h EMA50 (uptrend) AND 1d ATR ratio < 1.2 (low vol).
Short when RSI(2) > 90 AND price < 4h EMA50 (downtrend) AND 1d ATR ratio < 1.2.
Exit when RSI(2) crosses 50 OR volatility expands (ATR ratio > 1.5).
Uses 4h for trend direction, 1h for precise entry timing, 1d for volatility regime filter.
Target: 80-120 total trades over 4 years (20-30/year). RSI(2) captures extreme mean reversion,
4h EMA50 filters counter-trend trades, 1d ATR ratio avoids high volatility chop.
Works in bull markets (longs in uptrend pullbacks) and bear markets (shorts in downtrend bounces).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend filter (EMA50)
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMA50
    close_4h_series = pd.Series(close_4h)
    ema50_4h = close_4h_series.ewm(span=50, min_periods=50, adjust=False).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Get 1d data for volatility regime (ATR ratio)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ATR(10) and ATR(30) for regime
    high_1d_series = pd.Series(high_1d)
    low_1d_series = pd.Series(low_1d)
    close_1d_series = pd.Series(close_1d)
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr10 = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    atr30 = pd.Series(tr).rolling(window=30, min_periods=30).mean().values
    atr_ratio = atr10 / np.where(atr30 != 0, atr30, np.inf)
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # Calculate 1h RSI(2) for entry signals
    # RSI(2) = 100 - (100 / (1 + RS)), RS = avg_gain / avg_loss over 2 periods
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    # Wilder's smoothing (alpha = 1/period)
    avg_gain = pd.Series(gain).ewm(alpha=1/2, min_periods=2, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/2, min_periods=2, adjust=False).mean().values
    rs = avg_gain / np.where(avg_loss != 0, avg_loss, np.inf)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema50_4h_aligned[i]) or np.isnan(atr_ratio_aligned[i]) or 
            np.isnan(rsi[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema50 = ema50_4h_aligned[i]
        vol_regime = atr_ratio_aligned[i]
        rsi_val = rsi[i]
        
        if position == 0:
            # Long: RSI(2) < 10 (oversold) AND price > 4h EMA50 (uptrend) AND low volatility (ATR ratio < 1.2)
            if rsi_val < 10 and price > ema50 and vol_regime < 1.2:
                signals[i] = 0.20
                position = 1
            # Short: RSI(2) > 90 (overbought) AND price < 4h EMA50 (downtrend) AND low volatility (ATR ratio < 1.2)
            elif rsi_val > 90 and price < ema50 and vol_regime < 1.2:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: RSI(2) > 50 (mean reversion complete) OR volatility expands (ATR ratio > 1.5)
            if rsi_val > 50 or vol_regime > 1.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: RSI(2) < 50 (mean reversion complete) OR volatility expands (ATR ratio > 1.5)
            if rsi_val < 50 or vol_regime > 1.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_RSI2_MeanReversion_4hTrend_1dVolFilter"
timeframe = "1h"
leverage = 1.0