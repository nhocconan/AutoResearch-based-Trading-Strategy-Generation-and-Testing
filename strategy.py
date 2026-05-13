#!/usr/bin/env python3
# Hypothesis: 6h Williams Alligator with 12h/1d trend filter and volume confirmation.
# Williams Alligator: Jaw (EMA13, 8-bar offset), Teeth (EMA8, 5-bar offset), Lips (EMA5, 3-bar offset).
# Long when Lips > Teeth > Jaw (bullish alignment) AND price > 12h EMA34 AND volume > 1.5 * volume MA20.
# Short when Lips < Teeth < Jaw (bearish alignment) AND price < 12h EMA34 AND volume > 1.5 * volume MA20.
# Exit when Alligator alignment breaks or volume drops below average.
# Uses discrete position sizing (0.25) to limit trades to ~12-37/year and minimize fee churn.
# Designed to catch strong trends in both bull and bear markets with confirmation from higher timeframe trend and volume.

name = "6h_WilliamsAlligator_12hTrend_Volume_v1"
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
    volume = prices['volume'].values
    
    # Get 12h data for HTF trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate EMA(34) on 12h close for trend filter
    ema34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    # Williams Alligator components
    jaw = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values  # EMA13
    teeth = pd.Series(close).ewm(span=8, adjust=False, min_periods=8).mean().values   # EMA8
    lips = pd.Series(close).ewm(span=5, adjust=False, min_periods=5).mean().values    # EMA5
    
    # Apply Alligator offsets (shift right by specified bars)
    jaw = np.roll(jaw, 8)
    teeth = np.roll(teeth, 5)
    lips = np.roll(lips, 3)
    # Set first values to NaN to avoid invalid signals
    jaw[:8] = np.nan
    teeth[:5] = np.nan
    lips[:3] = np.nan
    
    # Volume confirmation: volume > 1.5 * 20-period volume moving average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma20)
    
    # Alligator alignment signals
    bullish_align = (lips > teeth) & (teeth > jaw)
    bearish_align = (lips < teeth) & (teeth < jaw)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after sufficient data for all indicators
        if np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or \
           np.isnan(ema34_12h_aligned[i]) or np.isnan(vol_ma20[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Bullish Alligator alignment AND price > 12h EMA34 AND volume confirmation
            if bullish_align[i] and close[i] > ema34_12h_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Bearish Alligator alignment AND price < 12h EMA34 AND volume confirmation
            elif bearish_align[i] and close[i] < ema34_12h_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Alligator alignment breaks OR volume drops below average
            if not bullish_align[i] or not volume_confirm[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Alligator alignment breaks OR volume drops below average
            if not bearish_align[i] or not volume_confirm[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals