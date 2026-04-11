#!/usr/bin/env python3
"""
6h_1d_williams_alligator_elder_ray_v1
Strategy: 6f Williams Alligator + Elder Ray (Bull/Bear Power) with 1d trend filter
Timeframe: 6h
Leverage: 1.0
Hypothesis: Go long when Alligator jaws (13-period SMMA) > teeth (8-period SMMA) > lips (5-period SMMA) and Elder Ray Bull Power > 0 on 6h, with 1d close above 200 EMA for trend alignment. Reverse for short. Uses SMMA (Smoothed Moving Average) to reduce whipsaw. Trend filter avoids counter-trend trades in strong markets. Designed for both bull and bear by following the dominant trend on 1d. Low-frequency design targets 15-30 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_williams_alligator_elder_ray_v1"
timeframe = "6h"
leverage = 1.0

def smma(source, length):
    """Smoothed Moving Average (SMMA) - also called RMA or Wilder's MA"""
    if length <= 0:
        return source.copy()
    result = np.full_like(source, np.nan, dtype=np.float64)
    # First value is simple average
    if len(source) >= length:
        result[length-1] = np.nanmean(source[:length])
    # Subsequent values: SMMA = (prev_smma * (length-1) + current) / length
    for i in range(length, len(source)):
        if not np.isnan(result[i-1]):
            result[i] = (result[i-1] * (length-1) + source[i]) / length
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load higher timeframe data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 6h Williams Alligator (SMMA based) ===
    # Jaws: 13-period SMMA of median price
    # Teeth: 8-period SMMA of median price  
    # Lips: 5-period SMMA of median price
    median_price = (high + low) / 2
    jaws = smma(median_price, 13)
    teeth = smma(median_price, 8)
    lips = smma(median_price, 5)
    
    # === 6h Elder Ray (Bull Power / Bear Power) ===
    # Bull Power = High - EMA(13)
    # Bear Power = Low - EMA(13)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False).mean().values
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # === 1d Trend Filter: EMA 200 ===
    close_1d = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Session filter: 00-23 UTC (6h bars cover full day, but avoid illiquid hours)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 0) & (hours <= 23)  # All hours for 6h
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid or outside session
        if (np.isnan(jaws[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(ema_200_1d_aligned[i]) or
            not in_session[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Alligator alignment: Jaws > Teeth > Lips (bullish) OR Jaws < Teeth < Lips (bearish)
        alligator_bull = jaws[i] > teeth[i] and teeth[i] > lips[i]
        alligator_bear = jaws[i] < teeth[i] and teeth[i] < lips[i]
        
        # Elder Ray: Bull Power > 0 and Bear Power < 0 for confirmation
        bull_power_pos = bull_power[i] > 0
        bear_power_neg = bear_power[i] < 0
        
        # 1d trend filter: price above/below 200 EMA
        price_above_200ema = close[i] > ema_200_1d_aligned[i]
        price_below_200ema = close[i] < ema_200_1d_aligned[i]
        
        # Long conditions: Alligator bullish + Bull Power positive + above 200 EMA
        long_signal = alligator_bull and bull_power_pos and price_above_200ema
        
        # Short conditions: Alligator bearish + Bear Power negative + below 200 EMA
        short_signal = alligator_bear and bear_power_neg and price_below_200ema
        
        # Exit when Alligator re-aligns (teeth crosses lips) or Elder Ray diverges
        exit_long = position == 1 and (teeth[i] < lips[i] or bull_power[i] <= 0)
        exit_short = position == -1 and (teeth[i] > lips[i] or bear_power[i] >= 0)
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

# Hypothesis: Go long when Alligator jaws (13-period SMMA) > teeth (8-period SMMA) > lips (5-period SMMA) and Elder Ray Bull Power > 0 on 6h, with 1d close above 200 EMA for trend alignment. Reverse for short. Uses SMMA (Smoothed Moving Average) to reduce whipsaw. Trend filter avoids counter-trend trades in strong markets. Designed for both bull and bear by following the dominant trend on 1d. Low-frequency design targets 15-30 trades/year to minimize fee drag.