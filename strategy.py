#!/usr/bin/env python3
# Hypothesis: 12h Williams Alligator with 1w EMA50 trend filter and ATR-based position sizing.
# Williams Alligator: Jaw (EMA13 of median price, 8-bar shift), Teeth (EMA8 of median price, 5-bar shift), Lips (EMA5 of median price, 3-bar shift).
# Long when Lips > Teeth > Jaw (bullish alignment) AND price > 1w EMA50.
# Short when Lips < Teeth < Jaw (bearish alignment) AND price < 1w EMA50.
# Position size scaled by ATR regime: 0.30 in high volatility (ATR20 > ATR60), 0.15 in low volatility.
# Uses discrete levels to minimize fee churn. Designed for 12-37 trades/year by requiring strong Alligator alignment and weekly trend filter.
# Works in bull markets via bullish Alligator alignment and in bear markets via bearish Alligator alignment.

name = "12h_WilliamsAlligator_1wTrend_ATR_Regime_v1"
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
    
    # Median price for Alligator
    median_price = (high + low) / 2.0
    
    # Williams Alligator components
    jaw = pd.Series(median_price).ewm(span=13, adjust=False, min_periods=13).mean().values
    jaw = np.roll(jaw, 8)  # 8-bar shift
    teeth = pd.Series(median_price).ewm(span=8, adjust=False, min_periods=8).mean().values
    teeth = np.roll(teeth, 5)  # 5-bar shift
    lips = pd.Series(median_price).ewm(span=5, adjust=False, min_periods=5).mean().values
    lips = np.roll(lips, 3)  # 3-bar shift
    
    # Get 1w data for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate EMA(50) on 1w close for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # ATR(20) and ATR(60) for volatility regime
    tr1 = np.maximum(high - low, np.absolute(high - np.roll(close, 1)))
    tr2 = np.absolute(low - np.roll(close, 1))
    tr = np.maximum(tr1, tr2)
    tr[0] = high[0] - low[0]  # first bar
    atr20 = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    atr60 = pd.Series(tr).rolling(window=60, min_periods=60).mean().values
    
    # Volatility regime: ATR20 > ATR60 (high volatility) -> full size, else half size
    vol_regime = atr20 > atr60
    position_size = np.where(vol_regime, 0.30, 0.15)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after sufficient data for all indicators
        if np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or \
           np.isnan(ema50_1w_aligned[i]) or np.isnan(atr20[i]) or np.isnan(atr60[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Bullish Alligator alignment (Lips > Teeth > Jaw) AND price > 1w EMA50
            if lips[i] > teeth[i] and teeth[i] > jaw[i] and close[i] > ema50_1w_aligned[i]:
                signals[i] = position_size[i]
                position = 1
            # SHORT: Bearish Alligator alignment (Lips < Teeth < Jaw) AND price < 1w EMA50
            elif lips[i] < teeth[i] and teeth[i] < jaw[i] and close[i] < ema50_1w_aligned[i]:
                signals[i] = -position_size[i]
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Alligator alignment breaks (Lips <= Teeth or Teeth <= Jaw) OR price < 1w EMA50 (trend break)
            if lips[i] <= teeth[i] or teeth[i] <= jaw[i] or close[i] < ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = position_size[i]
        elif position == -1:
            # EXIT SHORT: Alligator alignment breaks (Lips >= Teeth or Teeth >= Jaw) OR price > 1w EMA50 (trend break)
            if lips[i] >= teeth[i] or teeth[i] >= jaw[i] or close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -position_size[i]
    
    return signals