#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d EMA34 trend filter and ATR-based position sizing
# Long when Bull Power > 0 AND close > 1d EMA34 (uptrend confirmation)
# Short when Bear Power < 0 AND close < 1d EMA34 (downtrend confirmation)
# Exit when power crosses zero (mean reversion to equilibrium)
# Uses ATR(14) for dynamic position sizing: 0.30 * (ATR/price) capped at 0.30
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe
# Elder Ray measures trend strength via EMA13 relative to high/low, effective in both bull/bear markets
# 1d EMA34 provides higher-timeframe trend filter to avoid counter-trend whipsaws

name = "6h_ElderRay_1dEMA34_ATR_Volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    close_series = pd.Series(close)
    ema13 = close_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Get 1d data ONCE before loop for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA34
    close_1d_series = pd.Series(close_1d)
    ema34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align HTF indicators to 6h timeframe (wait for completed HTF bar)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate ATR(14) for dynamic position sizing
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar has no previous close
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Dynamic position size: 0.30 * (ATR/price) capped at 0.30
    # This reduces size during high volatility periods
    atr_ratio = np.where(close > 0, atr14 / close, 0)
    position_size = np.minimum(0.30, 0.30 * atr_ratio * 10)  # Scale to get meaningful size
    position_size = np.where(position_size < 0.05, 0.0, position_size)  # Minimum size threshold
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(position_size[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Elder Ray signals with trend filter
            # Long: Bull Power > 0 AND uptrend
            if bull_power[i] > 0 and close[i] > ema34_1d_aligned[i]:
                signals[i] = position_size[i]
                position = 1
            # Short: Bear Power < 0 AND downtrend
            elif bear_power[i] < 0 and close[i] < ema34_1d_aligned[i]:
                signals[i] = -position_size[i]
                position = -1
        elif position == 1:
            # Exit long: Bull Power crosses below zero (mean reversion)
            if bull_power[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = position_size[i]
        elif position == -1:
            # Exit short: Bear Power crosses above zero (mean reversion)
            if bear_power[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -position_size[i]
    
    return signals