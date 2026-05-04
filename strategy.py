#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + Elder Ray (Bull/Bear Power) with 1d ADX25 regime filter
# Williams Alligator defines trend via SMAs (Jaw=13, Teeth=8, Lips=5) - long when aligned up, short when aligned down
# Elder Ray measures bull/bear power: Bull Power = High - EMA13, Bear Power = EMA13 - Low
# Only take Alligator signals when Elder Ray confirms strength (Bull Power > 0 for longs, Bear Power > 0 for shorts)
# 1d ADX > 25 ensures we only trade in trending markets, avoiding whipsaws in ranges
# This combination reduces false signals while capturing strong trends in both bull and bear markets
# Target: 12-37 trades/year (50-150 total over 4 years) with discrete sizing to minimize fee drag

name = "6h_WilliamsAlligator_ElderRay_1dADX25_Regime"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ADX regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ADX (14-period) for regime filtering
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])  # Align with original indices
        
        # Directional Movement
        up_move = high[1:] - high[:-1]
        down_move = low[:-1] - low[1:]
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        plus_dm = np.concatenate([[np.nan], plus_dm])
        minus_dm = np.concatenate([[np.nan], minus_dm])
        
        # Smoothed TR, +DM, -DM using Wilder's smoothing (alpha = 1/period)
        alpha = 1.0 / period
        tr_smooth = np.full_like(tr, np.nan)
        plus_dm_smooth = np.full_like(plus_dm, np.nan)
        minus_dm_smooth = np.full_like(minus_dm, np.nan)
        
        # First value: simple average
        if len(tr) >= period + 1:
            tr_smooth[period] = np.nansum(tr[1:period+1])
            plus_dm_smooth[period] = np.nansum(plus_dm[1:period+1])
            minus_dm_smooth[period] = np.nansum(minus_dm[1:period+1])
            
            # Subsequent values: Wilder smoothing
            for i in range(period + 1, len(tr)):
                tr_smooth[i] = tr_smooth[i-1] - (tr_smooth[i-1] / period) + tr[i]
                plus_dm_smooth[i] = plus_dm_smooth[i-1] - (plus_dm_smooth[i-1] / period) + plus_dm[i]
                minus_dm_smooth[i] = minus_dm_smooth[i-1] - (minus_dm_smooth[i-1] / period) + minus_dm[i]
        
        # Directional Indicators
        plus_di = 100 * plus_dm_smooth / tr_smooth
        minus_di = 100 * minus_dm_smooth / tr_smooth
        
        # DX and ADX
        dx = np.full_like(tr, np.nan)
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        
        adx = np.full_like(tr, np.nan)
        if len(tr) >= 2 * period + 1:
            adx[2*period] = np.nanmean(dx[period+1:2*period+1])
            for i in range(2*period + 1, len(tr)):
                adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_1d_shifted = np.roll(adx_1d, 1)  # Use prior completed 1d bar
    adx_1d_shifted[0] = np.nan
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d_shifted)
    
    # Williams Alligator on 6h timeframe: SMAs of median price
    # Median price = (High + Low) / 2
    median_price = (high + low) / 2
    
    # Jaw: 13-period SMA, shifted 8 bars
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().values
    jaw = np.roll(jaw, 8)
    jaw[:8] = np.nan
    
    # Teeth: 8-period SMA, shifted 5 bars
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().values
    teeth = np.roll(teeth, 5)
    teeth[:5] = np.nan
    
    # Lips: 5-period SMA, shifted 3 bars
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().values
    lips = np.roll(lips, 3)
    lips[:3] = np.nan
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = EMA13 - Low
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = ema13 - low
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup period for Alligator
        # Skip if any value is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(adx_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Regime filter: only trade when 1d ADX > 25 (trending market)
        if adx_1d_aligned[i] <= 25:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Alligator aligned up (Lips > Teeth > Jaw) AND Bull Power > 0
            if lips[i] > teeth[i] and teeth[i] > jaw[i] and bull_power[i] > 0:
                signals[i] = 0.25
                position = 1
            # Short conditions: Alligator aligned down (Lips < Teeth < Jaw) AND Bear Power > 0
            elif lips[i] < teeth[i] and teeth[i] < jaw[i] and bear_power[i] > 0:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator alignment breaks OR Bull Power becomes negative
            if lips[i] <= teeth[i] or teeth[i] <= jaw[i] or bull_power[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator alignment breaks OR Bear Power becomes negative
            if lips[i] >= teeth[i] or teeth[i] >= jaw[i] or bear_power[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals