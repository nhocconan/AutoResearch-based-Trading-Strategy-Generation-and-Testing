#!/usr/bin/env python3
# 4h_RSI_Extremes_TrendFilter_Volume
# Hypothesis: RSI extremes (RSI < 25 for long, RSI > 75 for short) capture overbought/oversold conditions.
# Trend filter: 1d EMA50 ensures trades align with higher timeframe trend to avoid counter-trend whipsaws.
# Volume confirmation: current volume > 1.5 * 20-period average volume filters low-conviction moves.
# Designed to work in both bull and bear markets by only trading in direction of 1d trend.
# Uses discrete position sizing (0.25) to limit drawdown and minimize trade frequency.

name = "4h_RSI_Extremes_TrendFilter_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # RSI(14) on 4h close
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # Volume confirmation: current volume > 1.5 * 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(rsi[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Get aligned 1d close for trend filter
        close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
        close_1d_current = close_1d_aligned[i]
        
        trend_up = close_1d_current > ema50_1d_aligned[i]
        trend_down = close_1d_current < ema50_1d_aligned[i]
        
        vol_confirm = volume_confirm[i]
        
        if position == 0:
            # LONG: RSI oversold (<25) AND 1d uptrend AND volume confirmation
            if rsi[i] < 25 and trend_up and vol_confirm:
                signals[i] = 0.25
                position = 1
            # SHORT: RSI overbought (>75) AND 1d downtrend AND volume confirmation
            elif rsi[i] > 75 and trend_down and vol_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: RSI returns to neutral (>45) OR trend weakens
            if rsi[i] > 45 or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: RSI returns to neutral (<55) OR trend weakens
            if rsi[i] < 55 or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals