#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla H3/L3 breakout with 1d volume-weighted trend filter.
- Long when price breaks above Camarilla H3 AND 1d VWAP > 1d EMA34 (bullish regime)
- Short when price breaks below Camarilla L3 AND 1d VWAP < 1d EMA34 (bearish regime)
- Fixed position size 0.25 to limit trade frequency and fee drag
- Exit on opposite Camarilla breakout or regime change (VWAP/EMA34 crossover)
- Uses 12h primary with 1d HTF to target 12-37 trades/year (50-150 total over 4 years)
- Camarilla levels provide institutional support/resistance; VWAP/EMA34 confirms trend
- Designed to work in both bull (breakouts) and bear (mean reversion at extremes) markets
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
    
    # Camarilla levels from previous 12h bar
    # H3 = close + 1.1*(high-low)/4
    # L3 = close - 1.1*(high-low)/4
    # Use shift(1) to ensure we only use completed 12h bar
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close[0] = np.nan
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    
    rang = prev_high - prev_low
    H3 = prev_close + 1.1 * rang / 4
    L3 = prev_close - 1.1 * rang / 4
    
    # Get 1d data ONCE before loop for regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d VWAP and EMA34
    vwap_1d = (df_1d['close'] * df_1d['volume']).expanding().sum() / df_1d['volume'].expanding().sum()
    vwap_1d = vwap_1d.values
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d indicators to 12h timeframe
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Regime: bullish if VWAP > EMA34, bearish if VWAP < EMA34
    bullish_regime = vwap_1d_aligned > ema_34_1d_aligned
    bearish_regime = vwap_1d_aligned < ema_34_1d_aligned
    
    # Fixed position size to limit trades and control drawdown
    position_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 34  # EMA34 needs 34 periods
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(H3[i]) or np.isnan(L3[i]) or 
            np.isnan(vwap_1d_aligned[i]) or np.isnan(ema_34_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above H3 AND bullish regime
            if close[i] > H3[i] and bullish_regime[i]:
                signals[i] = position_size
                position = 1
            # Short: break below L3 AND bearish regime
            elif close[i] < L3[i] and bearish_regime[i]:
                signals[i] = -position_size
                position = -1
        elif position == 1:
            # Long exit: break below L3 OR regime turns bearish
            if close[i] < L3[i] or bearish_regime[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = position_size
        elif position == -1:
            # Short exit: break above H3 OR regime turns bullish
            if close[i] > H3[i] or bullish_regime[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_Camarilla_H3L3_1dVWAP_EMA34_Regime_v1"
timeframe = "12h"
leverage = 1.0