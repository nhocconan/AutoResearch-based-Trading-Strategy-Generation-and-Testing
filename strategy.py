#!/usr/bin/env python3
# 4h_Williams_Alligator_ElderRay_Trend
# Hypothesis: On 4h timeframe, use Williams Alligator (3 SMAs) to determine trend direction and avoid chop.
# Combine with Elder Ray (bull/bear power from EMA13) to confirm trend strength.
# Enter long when price > Alligator Jaw, Bull Power > 0, and Bear Power < 0 with volume confirmation.
# Enter short when price < Alligator Jaw, Bear Power > 0, and Bull Power < 0 with volume confirmation.
# Exit when Elder Ray signals reverse or price crosses the Alligator Teeth.
# Uses 1-day trend filter to avoid counter-trend trades, targeting 20-40 trades/year for low friction.
# Works in bull via strong bull power and in bear via strong bear power with trend alignment.

name = "4h_Williams_Alligator_ElderRay_Trend"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Williams Alligator: 3 SMAs (Jaw=13, Teeth=8, Lips=5)
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().values
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().values
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Volume confirmation: current volume > 1.3 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.3 * vol_ma)
    
    # 1-day trend filter: EMA50
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    daily_close = df_1d['close'].values
    daily_ema50 = pd.Series(daily_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    daily_ema50_aligned = align_htf_to_ltf(prices, df_1d, daily_ema50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(vol_ma[i]) or np.isnan(daily_ema50_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Alligator condition: price > Jaw for uptrend, price < Jaw for downtrend
        price_above_jaw = close[i] > jaw[i]
        price_below_jaw = close[i] < jaw[i]
        
        # Elder Ray condition: Bull Power > 0 and Bear Power < 0 for long,
        # Bear Power > 0 and Bull Power < 0 for short
        bull_strong = bull_power[i] > 0 and bear_power[i] < 0
        bear_strong = bear_power[i] > 0 and bull_power[i] < 0
        
        vol_confirm = volume_confirm[i]
        daily_trend = daily_ema50_aligned[i]
        
        if position == 0:
            # LONG: Price above Jaw, Bull Power positive, Bear Power negative, volume confirmation, daily uptrend
            if price_above_jaw and bull_strong and vol_confirm and close[i] > daily_trend:
                signals[i] = 0.25
                position = 1
            # SHORT: Price below Jaw, Bear Power positive, Bull Power negative, volume confirmation, daily downtrend
            elif price_below_jaw and bear_strong and vol_confirm and close[i] < daily_trend:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below Teeth or Elder Ray weakens
            if close[i] < teeth[i] or bull_power[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above Teeth or Elder Ray weakens
            if close[i] > teeth[i] or bear_power[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals