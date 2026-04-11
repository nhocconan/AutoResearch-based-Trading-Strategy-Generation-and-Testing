#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_camarilla_breakout_volume_trend_v5"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return signals
    
    # Calculate 1d Camarilla pivot levels
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Previous day's values for pivot calculation
    prev_close = np.roll(close_1d, 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    
    # Calculate pivot point
    pp = (prev_high + prev_low + prev_close) / 3.0
    
    # Calculate Camarilla levels
    range_ = prev_high - prev_low
    resistance_4 = pp + (range_ * 1.1 / 2)  # R4
    resistance_3 = pp + (range_ * 1.1 / 4)  # R3
    support_3 = pp - (range_ * 1.1 / 4)     # S3
    support_4 = pp - (range_ * 1.1 / 2)     # S4
    
    # Shift by 1 to use only completed 1d bars
    resistance_4 = np.roll(resistance_4, 1)
    resistance_3 = np.roll(resistance_3, 1)
    support_3 = np.roll(support_3, 1)
    support_4 = np.roll(support_4, 1)
    pp = np.roll(pp, 1)
    
    resistance_4[0] = np.nan
    resistance_3[0] = np.nan
    support_3[0] = np.nan
    support_4[0] = np.nan
    pp[0] = np.nan
    
    # Align 1d Camarilla levels to 4h timeframe
    resistance_4_aligned = align_htf_to_ltf(prices, df_1d, resistance_4)
    resistance_3_aligned = align_htf_to_ltf(prices, df_1d, resistance_3)
    support_3_aligned = align_htf_to_ltf(prices, df_1d, support_3)
    support_4_aligned = align_htf_to_ltf(prices, df_1d, support_4)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    
    # Calculate 4h EMA20 for trend filter
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(resistance_4_aligned[i]) or np.isnan(resistance_3_aligned[i]) or
            np.isnan(support_3_aligned[i]) or np.isnan(support_4_aligned[i]) or
            np.isnan(pp_aligned[i]) or np.isnan(ema_20[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        volume_current = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume confirmation
        volume_confirmed = volume_current > 1.5 * vol_ma
        
        # Long conditions: Close > S3 AND price > EMA20 (uptrend) with volume
        long_signal = volume_confirmed and (price_close > support_3_aligned[i]) and (price_close > ema_20[i])
        
        # Short conditions: Close < R3 AND price < EMA20 (downtrend) with volume
        short_signal = volume_confirmed and (price_close < resistance_3_aligned[i]) and (price_close < ema_20[i])
        
        # Exit when price crosses back through pivot point
        exit_long = position == 1 and price_close < pp_aligned[i]
        exit_short = position == -1 and price_close > pp_aligned[i]
        
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
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

# Hypothesis: Camarilla breakout with volume and EMA20 trend filter on 4h.
# Uses daily Camarilla pivot levels (S3, R3) as entry triggers and pivot point as exit.
# Enters long when price closes above S3 with volume confirmation (>1.5x average) and
# price above 4h EMA20 (uptrend). Enters short when price closes below R3 with volume
# confirmation and price below 4h EMA20 (downtrend). Exits when price crosses back
# through the daily pivot point. This strategy captures breakouts from key levels
# while avoiding counter-trend trades. The volume filter ensures participation from
# market actors. Target: 75-200 total trades over 4 years (19-50/year) to minimize
# fee drag on 4h timeframe. Designed to work in both bull and bear markets by
# following the trend defined by EMA20.