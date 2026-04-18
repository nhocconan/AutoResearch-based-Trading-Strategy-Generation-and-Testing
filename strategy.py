#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d RSI Extreme Reversal with 1w Trend Filter
# RSI extremes (>70 for overbought, <30 for oversold) indicate short-term exhaustion.
# Weekly trend filter (price above/below weekly EMA50) ensures we trade with the higher timeframe trend.
# This captures mean-reversion within the prevailing trend, reducing whipsaw.
# Works in both bull and bear markets by filtering counter-trend signals.
# Target: 10-25 trades/year (40-100 total over 4 years) to minimize fee drag.
name = "1d_RSI_Extreme_Reversal_1wTrend"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate RSI(14) on daily closes
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    # Pad RSI to align with close (RSI starts at index 14)
    rsi_padded = np.full(n, np.nan)
    rsi_padded[14:] = rsi
    
    # Calculate EMA50 on 1w close for trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for RSI and EMA calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if np.isnan(rsi_padded[i]) or np.isnan(ema_50_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        rsi_val = rsi_padded[i]
        ema_val = ema_50_1w_aligned[i]
        
        if position == 0:
            # Long: RSI oversold (<30) AND price above weekly EMA50 (uptrend)
            if rsi_val < 30 and close_val > ema_val:
                signals[i] = 0.25
                position = 1
            # Short: RSI overbought (>70) AND price below weekly EMA50 (downtrend)
            elif rsi_val > 70 and close_val < ema_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: RSI returns to neutral (>50) or trend change (price below weekly EMA50)
            if rsi_val > 50 or close_val < ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI returns to neutral (<50) or trend change (price above weekly EMA50)
            if rsi_val < 50 or close_val > ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals