#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h_1w_1d_trix_momentum_v1
# Uses TRIX (12-period triple EMA) on weekly timeframe for trend direction,
# combined with daily RSI(14) for momentum and volume confirmation on 6h chart.
# Long when weekly TRIX > 0, daily RSI < 30 (oversold), and volume > 1.5x 20-period average.
# Short when weekly TRIX < 0, daily RSI > 70 (overbought), and volume confirmation.
# Exits when RSI returns to neutral zone (40-60) or TRIX changes sign.
# Designed for low trade frequency (target: 15-25 trades/year) to minimize fee drag.
# Works in trending markets via TRIX direction and in ranging markets via RSI mean reversion.

name = "6h_1w_1d_trix_momentum_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for TRIX calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:  # Need enough data for triple EMA
        return np.zeros(n)
    
    # Calculate TRIX on weekly close
    close_1w = df_1w['close'].values
    
    # Triple EMA: EMA(EMA(EMA(close)))
    ema1 = pd.Series(close_1w).ewm(span=12, adjust=False, min_periods=12).mean()
    ema2 = ema1.ewm(span=12, adjust=False, min_periods=12).mean()
    ema3 = ema2.ewm(span=12, adjust=False, min_periods=12).mean()
    
    # TRIX = (EMA3 - previous EMA3) / previous EMA3 * 100
    trix_raw = (ema3 - ema3.shift(1)) / ema3.shift(1) * 100
    trix = trix_raw.fillna(0).values  # Handle NaN from shift
    
    # Align weekly TRIX to 6h timeframe
    trix_aligned = align_htf_to_ltf(prices, df_1w, trix)
    
    # Get daily data for RSI calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate RSI on daily close
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing (equivalent to EMA with alpha=1/14)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)  # Avoid division by zero
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.fillna(50).values  # Neutral RSI when not enough data
    
    # Align daily RSI to 6h timeframe
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi_values)
    
    # Volume confirmation: volume > 1.5 * 20-period average (6h timeframe)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # start after warmup
        # Skip if data not ready
        if np.isnan(trix_aligned[i]) or np.isnan(rsi_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Require volume confirmation for new entries
        if not vol_confirm[i]:
            # Hold current position if volume filter fails
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Long signal: weekly TRIX positive (uptrend), daily RSI oversold (<30)
        if trix_aligned[i] > 0 and rsi_aligned[i] < 30 and position != 1:
            position = 1
            signals[i] = 0.25
        # Short signal: weekly TRIX negative (downtrend), daily RSI overbought (>70)
        elif trix_aligned[i] < 0 and rsi_aligned[i] > 70 and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit conditions: RSI returns to neutral zone (40-60) or TRIX changes sign
        elif position == 1 and (rsi_aligned[i] >= 40 or trix_aligned[i] <= 0):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (rsi_aligned[i] <= 60 or trix_aligned[i] >= 0):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals