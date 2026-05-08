#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d RSI(14) mean reversion with volume filter and 1w trend filter.
# Long when RSI < 30 (oversold) AND 1w EMA40 rising AND volume > 1.5x 20-period average.
# Short when RSI > 70 (overbought) AND 1w EMA40 falling AND volume > 1.5x 20-period average.
# Exit when RSI crosses back to neutral (40-60 range) or opposite extreme.
# Uses 1d timeframe for mean reversion entries with 1w trend filter to avoid counter-trend trades.
# Volume confirmation filters out low-liquidity false signals.
# Target: 15-25 trades/year (60-100 total over 4 years) to minimize fee drag.
# Works in both bull and bear markets by following the weekly trend direction.

name = "1d_RSI14_MeanRev_1wEMA40_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 40:
        return np.zeros(n)
    
    # RSI(14) calculation
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # 1w EMA40 for trend filter
    ema40_1w = pd.Series(df_1w['close']).ewm(span=40, adjust=False, min_periods=40).mean().values
    ema40_1w_aligned = align_htf_to_ltf(prices, df_1w, ema40_1w)
    
    # 1w EMA40 direction
    ema40_rising = np.zeros_like(ema40_1w_aligned, dtype=bool)
    ema40_falling = np.zeros_like(ema40_1w_aligned, dtype=bool)
    ema40_rising[1:] = ema40_1w_aligned[1:] > ema40_1w_aligned[:-1]
    ema40_falling[1:] = ema40_1w_aligned[1:] < ema40_1w_aligned[:-1]
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(14, 20)  # Sufficient warmup for RSI and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(rsi[i]) or np.isnan(ema40_1w_aligned[i]) or 
            np.isnan(ema40_rising[i]) or np.isnan(ema40_falling[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: RSI < 30, 1w EMA40 rising, volume filter
            long_cond = (rsi[i] < 30) and ema40_rising[i] and volume_filter[i]
            # Short conditions: RSI > 70, 1w EMA40 falling, volume filter
            short_cond = (rsi[i] > 70) and ema40_falling[i] and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: RSI crosses above 40 or RSI > 70 (overbought)
            if rsi[i] > 40 or rsi[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: RSI crosses below 60 or RSI < 30 (oversold)
            if rsi[i] < 60 or rsi[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals