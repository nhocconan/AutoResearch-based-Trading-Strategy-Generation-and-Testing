#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 12h ADX(14) > 25 trend filter and 6h volume > 2.0x 20-period volume MA.
# Long when price breaks above Donchian upper band AND 12h ADX > 25 (strong trend) AND 6h volume spike.
# Short when price breaks below Donchian lower band AND 12h ADX > 25 AND 6h volume spike.
# Exit when price retraces to Donchian middle band (20-period SMA of HL/2) OR ADX < 20 (trend weakens).
# Uses session filter (08-20 UTC) to avoid low-liquidity periods. Position size 0.25.
# Designed for 6h timeframe to achieve 50-150 total trades over 4 years (12-37/year) with strict entry conditions.
# Donchian channels provide objective breakout levels, 12h ADX filters for strong trending markets only, volume confirms participation.
# Works in both bull and bear markets by only trading breakouts in strong trends (ADX>25) when volume confirms.

name = "6h_Donchian20_12hADX25_VolumeSpike_Session"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 12h data for ADX trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:  # Need enough for ADX calculation
        return np.zeros(n)
    
    # Calculate 12h ADX(14) for trend strength
    # ADX requires +DI, -DI, and DX calculations
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = pd.Series(high_12h).diff().abs()
    tr2 = pd.Series(high_12h - pd.Series(close_12h).shift(1)).abs()
    tr3 = pd.Series(low_12h - pd.Series(close_12h).shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).values
    
    # +DM and -DM
    up_move = pd.Series(high_12h).diff()
    down_move = -pd.Series(low_12h).diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values (Wilder's smoothing)
    def wilders_smoothing(values, period):
        result = np.zeros_like(values)
        result[:period-1] = np.nan
        if len(values) >= period:
            result[period-1] = np.nansum(values[:period])
            for i in range(period, len(values)):
                result[i] = result[i-1] - (result[i-1] / period) + values[i]
        return result
    
    tr_14 = wilders_smoothing(tr, 14)
    plus_dm_14 = wilders_smoothing(plus_dm, 14)
    minus_dm_14 = wilders_smoothing(minus_dm, 14)
    
    # +DI and -DI
    plus_di = np.where(tr_14 != 0, (plus_dm_14 / tr_14) * 100, 0)
    minus_di = np.where(tr_14 != 0, (minus_dm_14 / tr_14) * 100, 0)
    
    # DX and ADX
    dx = np.where((plus_di + minus_di) != 0, np.abs((plus_di - minus_di) / (plus_di + minus_di)) * 100, 0)
    adx = wilders_smoothing(dx, 14)
    
    # Align 12h ADX to 6h timeframe
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx)
    
    # Calculate Donchian channels from 6h data (20-period)
    donchian_window = 20
    donchian_high = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    donchian_mid = (donchian_high + donchian_low) / 2  # Middle band
    
    # Calculate 6h volume 20-period MA for spike detection
    volume_ma_6h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(adx_12h_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(donchian_mid[i]) or 
            np.isnan(volume_ma_6h[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        
        # Volume spike condition: current 6h volume > 2.0x 20-period volume MA
        volume_spike = volume[i] > (volume_ma_6h[i] * 2.0)
        
        # Donchian breakout conditions
        breakout_up = high_val > donchian_high[i]  # Price breaks above upper band
        breakout_down = low_val < donchian_low[i]  # Price breaks below lower band
        
        # 12h ADX trend strength condition (ADX > 25 = strong trend)
        strong_trend = adx_12h_aligned[i] > 25
        weak_trend = adx_12h_aligned[i] < 20  # For exit condition
        
        if position == 0:
            # Long: Donchian breakout up AND strong trend AND volume spike AND session
            if breakout_up and strong_trend and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: Donchian breakout down AND strong trend AND volume spike AND session
            elif breakout_down and strong_trend and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price retraces to middle band OR trend weakens
            if close_val < donchian_mid[i] or weak_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price retraces to middle band OR trend weakens
            if close_val > donchian_mid[i] or weak_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals