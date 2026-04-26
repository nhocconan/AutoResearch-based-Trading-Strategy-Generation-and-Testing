#!/usr/bin/env python3
"""
6h_ElderRay_BullBearPower_1dRegime_v1
Hypothesis: Elder Ray (Bull Power = High - EMA13, Bear Power = EMA13 - Low) on 6h with 1d EMA34 trend regime. 
Long when Bull Power > 0 AND Bear Power < 0 AND price > 1d EMA34 (bullish regime with buying pressure).
Short when Bull Power < 0 AND Bear Power > 0 AND price < 1d EMA34 (bearish regime with selling pressure).
Uses ATR(14) stoploss (2.5x) and discrete position sizing (0.25) to limit drawdown.
Designed for low trade frequency (target 12-37/year) by requiring confluence of trend, momentum, and regime.
Works in both bull (breakouts with regime alignment) and bear (mean reversion within regime) via Elder Ray's dual power measurement.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA trend regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate EMA(34) on 1d for trend regime
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate EMA(13) on 6h for Elder Ray
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate ATR(14) for stoploss on 6h
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Elder Ray components
    bull_power = high - ema_13  # Bull Power = High - EMA13
    bear_power = ema_13 - low   # Bear Power = EMA13 - Low
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: max of 1d EMA(34), 6h EMA(13), ATR
    start_idx = max(34, 13, 14) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(ema_13[i]) or
            np.isnan(atr[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        close_val = close[i]
        ema_13_val = ema_13[i]
        bull_power_val = bull_power[i]
        bear_power_val = bear_power[i]
        regime_up = close_val > ema_34_1d_aligned[i]   # 1d uptrend regime
        regime_down = close_val < ema_34_1d_aligned[i]  # 1d downtrend regime
        
        if position == 0:
            # Long: Bull Power > 0 (buying pressure) AND Bear Power < 0 (weak selling) AND bullish regime
            long_signal = (bull_power_val > 0) and (bear_power_val < 0) and regime_up
            
            # Short: Bull Power < 0 (weak buying) AND Bear Power > 0 (selling pressure) AND bearish regime
            short_signal = (bull_power_val < 0) and (bear_power_val > 0) and regime_down
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: regime flips down OR price hits ATR stoploss OR Elder Ray divergence (weakening momentum)
            if (not regime_up) or (close_val < entry_price - 2.5 * atr[i]) or (bull_power_val <= 0):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: regime flips up OR price hits ATR stoploss OR Elder Ray divergence (weakening momentum)
            if (not regime_down) or (close_val > entry_price + 2.5 * atr[i]) or (bear_power_val <= 0):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_ElderRay_BullBearPower_1dRegime_v1"
timeframe = "6h"
leverage = 1.0