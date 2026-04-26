#!/usr/bin/env python3
"""
6h_ElderRay_1wTrend_RegimeFilter
Hypothesis: Combines Elder Ray (Bull/Bear Power) with weekly trend and volatility regime filter.
- Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
- Weekly trend: EMA34 on 1w close
- Volatility regime: ATR(14) ratio (current vs 50-period median) to filter choppy markets
- Entry: Long when Bull Power > 0 AND weekly uptrend AND low volatility regime
         Short when Bear Power < 0 AND weekly downtrend AND low volatility regime
- Exit: Opposite signal or volatility expansion (regime shift to high volatility)
Designed for 6h timeframe to capture medium-term swings with reduced noise.
Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.
Works in both bull (trend following) and bear (mean reversion in regimes) markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for HTF trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly EMA(34) for trend filter
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate Elder Ray components (need EMA13 of close)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13  # High - EMA13
    bear_power = low - ema_13   # Low - EMA13
    
    # ATR(14) for volatility measurement and regime filter
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    # ATR ratio: current ATR vs 50-period median (regime filter)
    atr_median = pd.Series(atr).rolling(window=50, min_periods=50).median().values
    atr_ratio = atr / atr_median  # >1 = expanding volatility, <1 = contracting
    
    # Align HTF indicators to 6h timeframe
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of EMA13, weekly EMA34, ATR(14), ATR median(50)
    start_idx = max(13, 34, 14, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_13[i]) or 
            np.isnan(bull_power[i]) or
            np.isnan(bear_power[i]) or
            np.isnan(ema_34_1w_aligned[i]) or
            np.isnan(atr[i]) or
            np.isnan(atr_median[i]) or
            np.isnan(atr_ratio[i])):
            # Hold current position
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        bull_val = bull_power[i]
        bear_val = bear_power[i]
        ema_34_1w_val = ema_34_1w_aligned[i]
        close_val = close[i]
        atr_ratio_val = atr_ratio[i]
        
        # Regime filter: low volatility environment (ATR ratio < 1.2)
        low_vol_regime = atr_ratio_val < 1.2
        high_vol_regime = atr_ratio_val > 1.5  # volatility expansion for exit
        
        if position == 0:
            # Long: Bull Power positive (buying pressure) AND weekly uptrend AND low volatility
            long_signal = (bull_val > 0) and \
                          (close_val > ema_34_1w_val) and \
                          low_vol_regime
            
            # Short: Bear Power negative (selling pressure) AND weekly downtrend AND low volatility
            short_signal = (bear_val < 0) and \
                           (close_val < ema_34_1w_val) and \
                           low_vol_regime
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: weekly downtrend OR volatility expansion OR Bear Power turns negative
            if (close_val < ema_34_1w_val) or high_vol_regime or (bear_val < 0):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: weekly uptrend OR volatility expansion OR Bull Power turns positive
            if (close_val > ema_34_1w_val) or high_vol_regime or (bull_val > 0):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_ElderRay_1wTrend_RegimeFilter"
timeframe = "6h"
leverage = 1.0