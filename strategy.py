#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h breakout of 12-hour Donchian channel (15-period) with volume confirmation and ADX trend filter.
# Uses 12h timeframe for structure to reduce noise and false breakouts. Volume > 1.5x average and ADX > 20
# ensure trades occur only in strong trending conditions. Target: 20-35 trades/year per symbol.
# Position size: 0.25 to manage risk during drawdowns.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12-hour Donchian Channels (15-period)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    vol_12h = df_12h['volume'].values
    
    # Upper band = 15-period high, Lower band = 15-period low
    upper_donchian = pd.Series(high_12h).rolling(window=15, min_periods=15).max().values
    lower_donchian = pd.Series(low_12h).rolling(window=15, min_periods=15).min().values
    
    # Calculate 12-hour ADX (14-period) for trend strength
    # True Range
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    dm_plus = np.where((high_12h - np.roll(high_12h, 1)) > (np.roll(low_12h, 1) - low_12h),
                       np.maximum(high_12h - np.roll(high_12h, 1), 0), 0)
    dm_minus = np.where((np.roll(low_12h, 1) - low_12h) > (high_12h - np.roll(high_12h, 1)),
                        np.maximum(np.roll(low_12h, 1) - low_12h, 0), 0)
    
    # Smooth TR, DM+ and DM- (14-period Wilder's smoothing)
    def wilder_smooth(data, period):
        result = np.zeros_like(data)
        if len(data) < period + 1:
            return result
        result[period-1] = np.mean(data[1:period+1])
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    tr14 = wilder_smooth(tr, 14)
    dm_plus_14 = wilder_smooth(dm_plus, 14)
    dm_minus_14 = wilder_smooth(dm_minus, 14)
    
    # Directional Indicators
    plus_di = 100 * dm_plus_14 / tr14
    minus_di = 100 * dm_minus_14 / tr14
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = np.zeros_like(dx)
    if len(dx) < 28:
        return np.zeros(n)
    adx[27] = np.mean(dx[14:28])  # First ADX at index 27 (after 2*14-1)
    for i in range(28, len(dx)):
        adx[i] = (adx[i-1] * 13 + dx[i]) / 14
    
    # Align 12h indicators to 4H timeframe
    upper_donchian_aligned = align_htf_to_ltf(prices, df_12h, upper_donchian)
    lower_donchian_aligned = align_htf_to_ltf(prices, df_12h, lower_donchian)
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    vol_ma_15_aligned = align_htf_to_ltf(prices, df_12h,
                                         pd.Series(vol_12h).rolling(window=15, min_periods=15).mean().values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # Start after warmup for indicators
        # Skip if data not ready
        if (np.isnan(upper_donchian_aligned[i]) or np.isnan(lower_donchian_aligned[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma_15_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current price and volume (4h close and 12h volume aligned)
        price_close = prices['close'].iloc[i]
        vol_12h_current = align_htf_to_ltf(prices, df_12h, vol_12h)[i]
        
        if position == 0:
            # Enter long: price breaks above upper Donchian + volume surge + ADX > 20 (trending)
            if (price_close > upper_donchian_aligned[i] and
                vol_12h_current > 1.5 * vol_ma_15_aligned[i] and
                adx_aligned[i] > 20):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below lower Donchian + volume surge + ADX > 20
            elif (price_close < lower_donchian_aligned[i] and
                  vol_12h_current > 1.5 * vol_ma_15_aligned[i] and
                  adx_aligned[i] > 20):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: price returns to middle of Donchian channel or volatility drops
            exit_signal = False
            
            if position == 1:
                # Exit long: price < middle band or ADX weakens
                middle_donchian = (upper_donchian_aligned[i] + lower_donchian_aligned[i]) / 2
                if (price_close < middle_donchian) or adx_aligned[i] < 15:
                    exit_signal = True
            elif position == -1:
                # Exit short: price > middle band or ADX weakens
                middle_donchian = (upper_donchian_aligned[i] + lower_donchian_aligned[i]) / 2
                if (price_close > middle_donchian) or adx_aligned[i] < 15:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_12hDonchianBreakout_Volume_ADX"
timeframe = "4h"
leverage = 1.0