#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R Mean Reversion + 1d ADX Regime + Volume Spike
# Williams %R identifies overbought/oversold conditions. In ranging markets (ADX<25),
# extreme readings (>80 for oversold, <20 for overbought) signal mean reversion.
# Volume spike (>2x 20-period EMA) confirms conviction. Uses 6h timeframe with
# discrete position sizing (0.25) to balance trade frequency and fee drag.
# Designed to work in both bull (trend continuation via ADX>25) and bear (mean reversion in ranges) markets.

name = "6h_WilliamsR_MeanReversion_1dADX_Regime_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for ADX regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need sufficient data for ADX calculation
        return np.zeros(n)
    
    # Calculate 1d ADX (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # Align with 1d index
    
    # Directional Movement
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Smoothed TR, +DM, -DM (using Wilder's smoothing = EMA with alpha=1/period)
    def WilderSmoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value: simple average
        result[period-1] = np.nanmean(data[1:period]) if np.any(~np.isnan(data[1:period])) else np.nan
        # Rest: Wilder smoothing
        for i in range(period, len(data)):
            if np.isnan(result[i-1]):
                result[i] = np.nan
            else:
                result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    atr_1d = WilderSmoothing(tr, 14)
    plus_di_1d = 100 * WilderSmoothing(plus_dm, 14) / atr_1d
    minus_di_1d = 100 * WilderSmoothing(minus_dm, 14) / atr_1d
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)
    adx_1d = WilderSmoothing(dx_1d, 14)
    
    # Align 1d ADX to 6h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Williams %R (14-period) on 6h timeframe
    def WilliamsR(high, low, close, period=14):
        highest_high = np.maximum.accumulate(high)
        lowest_low = np.minimum.accumulate(low)
        # For proper lookback, we need rolling window
        highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
        wr = -100 * (highest_high - close) / (highest_high - lowest_low)
        return wr
    
    wr_6h = WilliamsR(high, low, close, 14)
    
    # Volume confirmation: 20-period EMA of volume on 6h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(wr_6h[i]) or np.isnan(adx_1d_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 2.0 x 20-period EMA
        volume_confirm = volume[i] > (2.0 * vol_ema_20[i])
        
        if position == 0:
            # Long signal: Williams %R oversold (< -80) + volume + regime filter
            # In ranging market (ADX < 25): mean reversion from oversold
            # In trending market (ADX >= 25): only long if also above 50 (momentum)
            if wr_6h[i] < -80 and volume_confirm:
                if adx_1d_aligned[i] < 25:  # Ranging: pure mean reversion
                    signals[i] = 0.25
                    position = 1
                else:  # Trending: require additional momentum filter
                    if wr_6h[i] > -50:  # Momentum confirmation
                        signals[i] = 0.25
                        position = 1
            # Short signal: Williams %R overbought (> -20) + volume + regime filter
            elif wr_6h[i] > -20 and volume_confirm:
                if adx_1d_aligned[i] < 25:  # Ranging: pure mean reversion
                    signals[i] = -0.25
                    position = -1
                else:  # Trending: require additional momentum filter
                    if wr_6h[i] < -50:  # Momentum confirmation
                        signals[i] = -0.25
                        position = -1
        elif position == 1:
            # Exit long: Williams %R returns above -50 (mean reversion complete) OR ADX strong trending down
            if wr_6h[i] > -50 or (adx_1d_aligned[i] > 30 and wr_6h[i] < -70):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R returns below -50 OR ADX strong trending up
            if wr_6h[i] < -50 or (adx_1d_aligned[i] > 30 and wr_6h[i] > -30):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals