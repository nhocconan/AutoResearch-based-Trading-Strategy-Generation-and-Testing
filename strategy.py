#!/usr/bin/env python3
# Hypothesis: 6h Williams Alligator + Elder Ray combination with 1d trend filter.
# Long when: Green candle, Jaw < Teeth < Lips (Alligator aligned up), Bull Power > 0 (Elder Ray), and close > 1d EMA50.
# Short when: Red candle, Jaw > Teeth > Lips (Alligator aligned down), Bear Power < 0 (Elder Ray), and close < 1d EMA50.
# Exit on opposing signal. Uses 6h primary timeframe with 1d trend filter to avoid counter-trend trades.
# Williams Alligator identifies trend alignment, Elder Ray measures bull/bear power behind moves.
# Designed for low frequency (12-35 trades/year) to minimize fee drag while capturing sustained moves.

name = "6h_WilliamsAlligator_ElderRay_1dEMA50_v1"
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
    open_price = prices['open'].values
    
    # Williams Alligator (6h): Jaw=13, Teeth=8, Lips=5 SMAs shifted
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().values
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().values
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().values
    # Shift to avoid look-ahead (Alligator uses future data if not shifted)
    jaw = np.roll(jaw, 5)
    teeth = np.roll(teeth, 3)
    lips = np.roll(lips, 2)
    # Fill NaN from roll
    jaw[:5] = jaw[5] if not np.isnan(jaw[5]) else close[5] if len(close) > 5 else close[0]
    teeth[:3] = teeth[3] if not np.isnan(teeth[3]) else close[3] if len(close) > 3 else close[0]
    lips[:2] = lips[2] if not np.isnan(lips[2]) else close[2] if len(close) > 2 else close[0]
    
    # Elder Ray (6h): Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Get 1d data for EMA50 trend filter (MTF)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA50 on 1d close
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF arrays to 6h timeframe (wait for completed 1d bar)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after sufficient data for indicators
        # Skip if any required data is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(ema50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Determine candle color
        is_green = close[i] > open_price[i]
        is_red = close[i] < open_price[i]
        
        if position == 0:
            # LONG: Green candle + Alligator aligned up (Jaw < Teeth < Lips) + Bull Power > 0 + close > 1d EMA50
            if (is_green and jaw[i] < teeth[i] and teeth[i] < lips[i] and 
                bull_power[i] > 0 and close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Red candle + Alligator aligned down (Jaw > Teeth > Lips) + Bear Power < 0 + close < 1d EMA50
            elif (is_red and jaw[i] > teeth[i] and teeth[i] > lips[i] and 
                  bear_power[i] < 0 and close[i] < ema50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Any opposing condition (red candle or Alligator alignment down or Bear Power >= 0)
            if (is_red or jaw[i] >= teeth[i] or teeth[i] >= lips[i] or bear_power[i] >= 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Any opposing condition (green candle or Alligator alignment up or Bull Power <= 0)
            if (is_green or jaw[i] <= teeth[i] or teeth[i] <= lips[i] or bull_power[i] <= 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals