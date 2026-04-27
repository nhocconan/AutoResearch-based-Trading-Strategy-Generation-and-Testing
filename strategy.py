#!/usr/bin/env python3
"""
4h_RSI2_MeanReversion_12hTrend_Filter
Hypothesis: RSI(2) mean reversion works when aligned with 12h trend (EMA50) to avoid counter-trend trades.
            Uses volume confirmation to filter false signals. Designed for low trade frequency (<30/year)
            to minimize fee drag and work in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # 12h trend: EMA50
    ema50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # RSI(2) for mean reversion signals
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/2, adjust=False, min_periods=2).mean()
    avg_loss = loss.ewm(alpha=1/2, adjust=False, min_periods=2).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Volume confirmation: volume > 2.0 * 20-period average (less strict to avoid over-filtering)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for RSI(2) and volume MA
    start_idx = max(30, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if np.isnan(ema50_12h_aligned[i]) or np.isnan(rsi_values[i]):
            signals[i] = 0.0
            continue
        
        ema_trend = ema50_12h_aligned[i]
        rsi_val = rsi_values[i]
        vol_conf = vol_confirm[i]
        
        if position == 0:
            # Long: RSI(2) < 10 (oversold) + volume confirmation + uptrend (price > EMA)
            if rsi_val < 10 and vol_conf and close[i] > ema_trend:
                signals[i] = size
                position = 1
            # Short: RSI(2) > 90 (overbought) + volume confirmation + downtrend (price < EMA)
            elif rsi_val > 90 and vol_conf and close[i] < ema_trend:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: RSI(2) > 50 (mean reversion complete) or trend breakdown
            if rsi_val > 50 or close[i] < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: RSI(2) < 50 (mean reversion complete) or trend reversal
            if rsi_val < 50 or close[i] > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_RSI2_MeanReversion_12hTrend_Filter"
timeframe = "4h"
leverage = 1.0