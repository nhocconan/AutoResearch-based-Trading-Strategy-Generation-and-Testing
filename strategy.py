#!/usr/bin/env python3
"""
Hypothesis: 1h EMA(21) pullback strategy with 4h EMA(50) trend filter and 1d RSI(14) regime filter.
- Long: 4h EMA50 up (EMA50 > EMA50_prev) AND 1d RSI < 60 (not overbought) AND price pulls back to 1h EMA21 from above
- Short: 4h EMA50 down (EMA50 < EMA50_prev) AND 1d RSI > 40 (not oversold) AND price pulls back to 1h EMA21 from below
- Entry: Long when low <= EMA21 AND close > EMA21 (bullish bounce); Short when high >= EMA21 AND close < EMA21 (bearish rejection)
- Exit: Opposite pullback condition OR 4h EMA50 flips
- Uses 4h for trend direction and 1d for regime to avoid counter-trend trades in extreme conditions
- Designed for low trade frequency (15-35/year) to minimize fee drag on 1h timeframe
- Pullback entries provide better risk-reward than breakouts in ranging markets
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
    
    # 1h EMA21 for pullback entries
    ema_21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # 4h EMA50 for trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # 1d RSI14 for regime filter (avoid overbought/oversold extremes)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    delta = pd.Series(close_1d).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_14 = 100 - (100 / (1 + rs))
    rsi_14_aligned = align_htf_to_ltf(prices, df_1d, rsi_14.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(21, 50, 14)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_21[i]) or 
            np.isnan(ema_50_4h_aligned[i]) or
            np.isnan(rsi_14_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # 4h EMA50 trend direction (using prior bar to avoid look-ahead)
        ema50_up = ema_50_4h_aligned[i] > ema_50_4h_aligned[i-1]
        ema50_down = ema_50_4h_aligned[i] < ema_50_4h_aligned[i-1]
        
        # 1d RSI regime filter
        rsi_not_overbought = rsi_14_aligned[i] < 60
        rsi_not_oversold = rsi_14_aligned[i] > 40
        
        # 1h EMA21 pullback signals
        pullback_long = low[i] <= ema_21[i] and close[i] > ema_21[i]  # Bullish bounce off EMA21
        pullback_short = high[i] >= ema_21[i] and close[i] < ema_21[i]  # Bearish rejection at EMA21
        
        if position == 0:
            # Long: 4h EMA50 up AND RSI not overbought AND bullish pullback to EMA21
            if ema50_up and rsi_not_overbought and pullback_long:
                signals[i] = 0.20
                position = 1
            # Short: 4h EMA50 down AND RSI not oversold AND bearish pullback to EMA21
            elif ema50_down and rsi_not_oversold and pullback_short:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: Bearish pullback at EMA21 OR 4h EMA50 flips down
            if pullback_short or ema50_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: Bullish bounce at EMA21 OR 4h EMA50 flips up
            if pullback_long or ema50_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_EMA21_Pullback_4hEMA50_Trend_1dRSI_Regime"
timeframe = "1h"
leverage = 1.0