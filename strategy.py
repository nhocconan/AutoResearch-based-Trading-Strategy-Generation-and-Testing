#!/usr/bin/env python3
# Hypothesis: 12h Williams Alligator with 1d EMA34 trend filter and volatility-adjusted position sizing.
# Williams Alligator: Jaw (EMA13, 8-period smoothed), Teeth (EMA8, 5-period smoothed), Lips (EMA5, 3-period smoothed).
# Long when Lips > Teeth > Jaw (bullish alignment) AND price > 1d EMA34.
# Short when Lips < Teeth < Jaw (bearish alignment) AND price < 1d EMA34.
# Position size scaled by ATR regime: 0.25 in high volatility (ATR14 > ATR50), 0.15 in low volatility.
# Uses discrete levels to minimize fee churn. Designed for 12-37 trades/year by requiring strong Alligator alignment and daily trend filter.
# Works in bull markets via bullish Alligator alignment and in bear markets via bearish Alligator alignment.

name = "12h_Williams_Alligator_1dTrend_VolRegime_v1"
timeframe = "12h"
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
    
    # Get 1d data for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA(34) on 1d close for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Williams Alligator components
    # Jaw: EMA(13) smoothed by 8 periods
    jaw_raw = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    jaw = pd.Series(jaw_raw).ewm(span=8, adjust=False, min_periods=8).mean().values
    
    # Teeth: EMA(8) smoothed by 5 periods
    teeth_raw = pd.Series(close).ewm(span=8, adjust=False, min_periods=8).mean().values
    teeth = pd.Series(teeth_raw).ewm(span=5, adjust=False, min_periods=5).mean().values
    
    # Lips: EMA(5) smoothed by 3 periods
    lips_raw = pd.Series(close).ewm(span=5, adjust=False, min_periods=5).mean().values
    lips = pd.Series(lips_raw).ewm(span=3, adjust=False, min_periods=3).mean().values
    
    # ATR(14) and ATR(50) for volatility regime
    tr1 = np.maximum(high - low, np.absolute(high - np.roll(close, 1)))
    tr2 = np.absolute(low - np.roll(close, 1))
    tr = np.maximum(tr1, tr2)
    tr[0] = high[0] - low[0]  # first bar
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    
    # Volatility regime: ATR14 > ATR50 (high volatility) -> larger size, else smaller size
    vol_regime = atr14 > atr50
    position_size = np.where(vol_regime, 0.25, 0.15)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after sufficient data for all indicators
        if np.isnan(ema34_1d_aligned[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or \
           np.isnan(atr14[i]) or np.isnan(atr50[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Bullish Alligator alignment (Lips > Teeth > Jaw) AND price > 1d EMA34
            if lips[i] > teeth[i] and teeth[i] > jaw[i] and close[i] > ema34_1d_aligned[i]:
                signals[i] = position_size[i]
                position = 1
            # SHORT: Bearish Alligator alignment (Lips < Teeth < Jaw) AND price < 1d EMA34
            elif lips[i] < teeth[i] and teeth[i] < jaw[i] and close[i] < ema34_1d_aligned[i]:
                signals[i] = -position_size[i]
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Loss of bullish alignment OR price < 1d EMA34 (trend break)
            if not (lips[i] > teeth[i] and teeth[i] > jaw[i]) or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = position_size[i]
        elif position == -1:
            # EXIT SHORT: Loss of bearish alignment OR price > 1d EMA34 (trend break)
            if not (lips[i] < teeth[i] and teeth[i] < jaw[i]) or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -position_size[i]
    
    return signals