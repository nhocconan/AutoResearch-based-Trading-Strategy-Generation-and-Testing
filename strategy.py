#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Elder Ray Index with 1-day regime filter and 1-week trend strength
# Long when Bull Power > 0 AND Bear Power < 0 AND price > 1-day EMA50 (bullish regime)
# Short when Bear Power > 0 AND Bull Power < 0 AND price < 1-day EMA50 (bearish regime)
# Exit when Elder Ray signals weaken or reverse
# Stoploss at 2.5 * ATR(22)
# Position size: 0.25
# Uses 1-day EMA50 for regime filter and 1-week ATR for volatility scaling
# Target: 100-200 total trades over 4 years (25-50/year)

name = "6h_elder_ray_1d_regime_1w_atr_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 1-day data for regime filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1-week data for ATR-based volatility scaling
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 22:
        return np.zeros(n)
    
    # Calculate 1-day EMA50 for regime filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate 1-week ATR for volatility scaling
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr_1w = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1w = pd.Series(tr_1w).ewm(span=22, adjust=False, min_periods=22).mean().values
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_1w)
    
    # Calculate Elder Ray components (13-period EMA for power)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # ATR for stoploss (22-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=22, adjust=False, min_periods=22).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(atr_1w_aligned[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2.5 * ATR
            if close[i] < entry_price - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: Elder Ray turns bearish or regime changes
            elif bull_power[i] <= 0 or bear_power[i] >= 0 or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2.5 * ATR
            if close[i] > entry_price + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: Elder Ray turns bullish or regime changes
            elif bear_power[i] <= 0 or bull_power[i] >= 0 or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Elder Ray with regime filter
            # Bullish regime: price above 1-day EMA50
            bullish_regime = close[i] > ema50_1d_aligned[i]
            # Bearish regime: price below 1-day EMA50
            bearish_regime = close[i] < ema50_1d_aligned[i]
            
            # Strong bullish momentum: Bull Power > 0 AND Bear Power < 0
            bullish_momentum = bull_power[i] > 0 and bear_power[i] < 0
            # Strong bearish momentum: Bear Power > 0 AND Bull Power < 0
            bearish_momentum = bear_power[i] > 0 and bull_power[i] < 0
            
            # Long: bullish regime + bullish momentum
            if bullish_regime and bullish_momentum:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: bearish regime + bearish momentum
            elif bearish_regime and bearish_momentum:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals