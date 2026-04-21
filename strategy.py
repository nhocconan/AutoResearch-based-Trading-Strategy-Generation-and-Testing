#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h breakout of 1-day Bollinger Bands with volume confirmation and ADX trend filter.
# Works in bull markets (breakouts capture momentum) and bear markets (breakdowns capture downtrends).
# Uses tight entry conditions (volume > 1.5x average, ADX > 20) to limit trades to ~25-40/year.
# Position size: 0.25 to balance risk and return.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1-day Bollinger Bands (20-period, 2 std dev)
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    vol_1d = df_1d['volume'].values
    
    # Middle band = 20-day SMA
    sma_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    # Standard deviation
    std_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    
    # Calculate 1-day ADX (14-period) for trend strength
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    
    # Smooth TR, DM+ and DM- (14-period Wilder's smoothing)
    def wilder_smooth(data, period):
        result = np.zeros_like(data)
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
    adx[27] = np.mean(dx[14:28])  # First ADX at index 27 (after 2*14-1)
    for i in range(28, len(dx)):
        adx[i] = (adx[i-1] * 13 + dx[i]) / 14
    
    # Align 1D indicators to 4H timeframe
    upper_bb_aligned = align_htf_to_ltf(prices, df_1d, upper_bb)
    lower_bb_aligned = align_htf_to_ltf(prices, df_1d, lower_bb)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d,
                                         pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # Start after warmup for indicators
        # Skip if data not ready
        if (np.isnan(upper_bb_aligned[i]) or np.isnan(lower_bb_aligned[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma_20_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current price and volume (4h close and 1d volume aligned)
        price_close = prices['close'].iloc[i]
        vol_1d_current = align_htf_to_ltf(prices, df_1d, vol_1d)[i]
        
        if position == 0:
            # Enter long: price breaks above upper BB + volume surge + ADX > 20 (trending)
            if (price_close > upper_bb_aligned[i] and
                vol_1d_current > 1.5 * vol_ma_20_aligned[i] and
                adx_aligned[i] > 20):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below lower BB + volume surge + ADX > 20
            elif (price_close < lower_bb_aligned[i] and
                  vol_1d_current > 1.5 * vol_ma_20_aligned[i] and
                  adx_aligned[i] > 20):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: price returns to middle Bollinger Band or volatility drops
            exit_signal = False
            
            if position == 1:
                # Exit long: price < middle band or ADX weakens
                if (price_close < sma_20[-1] if len(sma_20) > 0 else 0) or adx_aligned[i] < 15:
                    exit_signal = True
            elif position == -1:
                # Exit short: price > middle band or ADX weakens
                if (price_close > sma_20[-1] if len(sma_20) > 0 else 0) or adx_aligned[i] < 15:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_BollingerBreakout_Volume_ADX"
timeframe = "4h"
leverage = 1.0