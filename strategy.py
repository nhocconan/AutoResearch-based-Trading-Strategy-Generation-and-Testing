#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + weekly regime filter
# - Bull Power = High - EMA13(close), Bear Power = EMA13(close) - Low
# - Long when Bull Power > 0 AND Bear Power < 0 AND weekly close > weekly EMA34 (bullish regime)
# - Short when Bear Power > 0 AND Bull Power < 0 AND weekly close < weekly EMA34 (bearish regime)
# - Uses 6h timeframe targeting 12-37 trades/year (50-150 total over 4 years)
# - Weekly regime filter ensures trading with higher timeframe trend
# - Discrete position sizing (0.25) to minimize fee churn
# - ATR-based stoploss: exit when price moves against position by 2.5x ATR(14)

name = "6h_1w_elder_ray_regime_atr_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Pre-compute weekly EMA(34) for regime filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(34, n):
        # Skip if weekly regime data is invalid
        if np.isnan(ema_34_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Calculate Elder Ray components using 6h data
        ema_13 = pd.Series(prices['close'].values).ewm(span=13, adjust=False, min_periods=13).mean().iloc[i]
        bull_power = prices['high'].iloc[i] - ema_13
        bear_power = ema_13 - prices['low'].iloc[i]
        
        if position == 1:  # Long position
            # Exit: ATR-based stoploss or regime turns bearish
            atr_14 = calculate_atr(prices['high'].values, prices['low'].values, prices['close'].values, 14)
            if (prices['close'].iloc[i] < entry_price - 2.5 * atr_14[i] or 
                prices['close'].iloc[i] < ema_34_1w_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: ATR-based stoploss or regime turns bullish
            atr_14 = calculate_atr(prices['high'].values, prices['low'].values, prices['close'].values, 14)
            if (prices['close'].iloc[i] > entry_price + 2.5 * atr_14[i] or 
                prices['close'].iloc[i] > ema_34_1w_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Elder Ray signals with weekly regime filter
            weekly_bullish = prices['close'].iloc[i] > ema_34_1w_aligned[i]
            weekly_bearish = prices['close'].iloc[i] < ema_34_1w_aligned[i]
            
            # Long: Bull Power positive AND Bear Power negative AND weekly bullish regime
            if bull_power > 0 and bear_power < 0 and weekly_bullish:
                position = 1
                entry_price = prices['close'].iloc[i]
                signals[i] = 0.25
            # Short: Bear Power positive AND Bull Power negative AND weekly bearish regime
            elif bear_power > 0 and bull_power < 0 and weekly_bearish:
                position = -1
                entry_price = prices['close'].iloc[i]
                signals[i] = -0.25
    
    return signals

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing (equivalent to RMA)"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr = np.zeros_like(tr)
    atr[period-1] = np.mean(tr[:period])
    for i in range(period, len(tr)):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    return atr