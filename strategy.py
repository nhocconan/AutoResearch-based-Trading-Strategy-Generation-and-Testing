#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + Elder Ray + Volume confirmation.
# Alligator identifies trend direction via SMAs (jaw, teeth, lips).
# Elder Ray measures bull/bear power relative to EMA13.
# Volume confirms conviction. Designed for low-frequency signals on 12h to avoid overtrading.
# Works in bull/bear via trend-following + mean-reversion hybrid logic.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily data for multi-timeframe analysis
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Williams Alligator on 12h: SMA(13,8,5) shifted
    jaw_period = 13
    teeth_period = 8
    lips_period = 5
    
    def sma(arr, period):
        res = np.full_like(arr, np.nan, dtype=np.float64)
        for i in range(period-1, len(arr)):
            res[i] = np.mean(arr[i-period+1:i+1])
        return res
    
    jaw = sma(close, jaw_period)
    teeth = sma(close, teeth_period)
    lips = sma(close, lips_period)
    
    # Shift as per Alligator: jaw by 8, teeth by 5, lips by 3
    jaw = np.roll(jaw, 8)
    teeth = np.roll(teeth, 5)
    lips = np.roll(lips, 3)
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    def ema(arr, period):
        res = np.full_like(arr, np.nan, dtype=np.float64)
        if len(arr) < period:
            return res
        multiplier = 2 / (period + 1)
        res[period-1] = np.mean(arr[:period])
        for i in range(period, len(arr)):
            res[i] = (arr[i] - res[i-1]) * multiplier + res[i-1]
        return res
    
    ema13 = ema(close, 13)
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Align daily trend filter: EMA21 on 1d
    close_1d = df_1d['close'].values
    ema21_1d = np.zeros(len(close_1d))
    if len(close_1d) >= 21:
        multiplier = 2 / (21 + 1)
        ema21_1d[20] = np.mean(close_1d[:21])
        for i in range(21, len(close_1d)):
            ema21_1d[i] = (close_1d[i] - ema21_1d[i-1]) * multiplier + ema21_1d[i-1]
    ema21_1d_aligned = align_htf_to_ltf(prices, df_1d, ema21_1d)
    
    # Volume confirmation: volume > 1.5x average volume (20-period)
    avg_volume = np.zeros(n)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(30, n):
        # Skip if any required data is not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(avg_volume[i]) or np.isnan(ema21_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        
        # Alligator conditions: teeth > jaws = uptrend, teeth < jaws = downtrend
        alligator_long = teeth[i] > jaw[i]
        alligator_short = teeth[i] < jaw[i]
        
        # Elder Ray: bull power > 0 and bear power < 0 for confirmation
        bull_power_pos = bull_power[i] > 0
        bear_power_neg = bear_power[i] < 0
        
        # Volume confirmation
        volume_confirm = vol > 1.5 * avg_vol
        
        # Daily trend filter: price above/below daily EMA21
        price_above_daily_ema = price > ema21_1d_aligned[i]
        price_below_daily_ema = price < ema21_1d_aligned[i]
        
        if position == 0:
            # Long: Alligator uptrend + bull power positive + volume + price above daily EMA
            if (alligator_long and bull_power_pos and volume_confirm and price_above_daily_ema):
                position = 1
                signals[i] = position_size
            # Short: Alligator downtrend + bear power negative + volume + price below daily EMA
            elif (alligator_short and bear_power_neg and volume_confirm and price_below_daily_ema):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Alligator reverses OR Elder Ray weakens
            if (not alligator_long or not bull_power_pos or vol < 0.5 * avg_vol):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Alligator reverses OR Elder Ray weakens
            if (not alligator_short or not bear_power_neg or vol < 0.5 * avg_vol):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1d_Williams_Alligator_ElderRay_Volume"
timeframe = "12h"
leverage = 1.0