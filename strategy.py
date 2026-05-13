#!/usr/bin/env python3
# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1w EMA34 trend filter and ATR-based regime filter.
# Long when Bull Power > 0 AND price > 1w EMA34 AND ATR(14) < ATR(50) (low volatility regime).
# Short when Bear Power < 0 AND price < 1w EMA34 AND ATR(14) < ATR(50).
# Exit when power reverses OR ATR(14) > ATR(50) (high volatility regime exit).
# Uses discrete position sizing (0.25) to minimize fee churn. Target: 12-37 trades/year.
# Works in bull markets via trend continuation and in bear markets via faded rallies in low vol regimes.

name = "6h_ElderRay_BullBearPower_1wTrend_ATRRegime_v1"
timeframe = "6h"
leverage = 1.0

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
    
    # Get 1w data for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate EMA(34) on 1w close for trend filter
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Calculate ATR(14) and ATR(50) for regime filter
    tr1 = np.maximum(high - low, np.abs(high - np.roll(close, 1)))
    tr2 = np.maximum(tr1, np.abs(low - np.roll(close, 1)))
    tr2[0] = high[0] - low[0]  # first bar
    atr14 = pd.Series(tr2).rolling(window=14, min_periods=14).mean().values
    atr50 = pd.Series(tr2).rolling(window=50, min_periods=50).mean().values
    
    # Calculate Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after sufficient data for all indicators
        if np.isnan(ema34_1w_aligned[i]) or np.isnan(ema13[i]) or np.isnan(atr14[i]) or np.isnan(atr50[i]):
            signals[i] = 0.0
            continue
        
        # Regime filter: low volatility (ATR14 < ATR50)
        low_vol_regime = atr14[i] < atr50[i]
        
        if position == 0:
            # LONG: Bull Power > 0 AND price > 1w EMA34 AND low volatility regime
            if bull_power[i] > 0 and close[i] > ema34_1w_aligned[i] and low_vol_regime:
                signals[i] = 0.25
                position = 1
            # SHORT: Bear Power < 0 AND price < 1w EMA34 AND low volatility regime
            elif bear_power[i] < 0 and close[i] < ema34_1w_aligned[i] and low_vol_regime:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Bull Power <= 0 OR price < 1w EMA34 OR high volatility regime
            if bull_power[i] <= 0 or close[i] < ema34_1w_aligned[i] or not low_vol_regime:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Bear Power >= 0 OR price > 1w EMA34 OR high volatility regime
            if bear_power[i] >= 0 or close[i] > ema34_1w_aligned[i] or not low_vol_regime:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals