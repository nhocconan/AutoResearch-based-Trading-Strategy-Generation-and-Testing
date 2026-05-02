#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Power with 1d ADX regime filter and volume confirmation
# Elder Ray measures bull/bear power via EMA13: Bull Power = High - EMA13, Bear Power = Low - EMA13
# In bull markets (ADX>25 + +DI>-DI), we buy when Bull Power turns positive with volume
# In bear markets (ADX>25 + -DI>+DI), we sell when Bear Power turns negative with volume
# In ranging markets (ADX<20), we fade extremes: short at Bull Power peaks, long at Bear Power troughs
# Designed for 50-150 total trades over 4 years (12-37/year) on 6h timeframe.
# Works in bull/bear/range by adapting logic to regime via ADX.

name = "6h_ElderRay_Power_ADX_Regime_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA13 for Elder Ray Power
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate 1d ADX for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.max([high_1d[0] - low_1d[0], 0])], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed TR, +DM, -DM (Wilder's smoothing)
    def wilders_smoothing(values, period):
        result = np.full_like(values, np.nan)
        if len(values) < period:
            return result
        result[period-1] = np.nansum(values[:period])
        for i in range(period, len(values)):
            result[i] = result[i-1] - (result[i-1] / period) + values[i]
        return result
    
    atr = wilders_smoothing(tr, 14)
    plus_di = 100 * wilders_smoothing(plus_dm, 14) / atr
    minus_di = 100 * wilders_smoothing(minus_dm, 14) / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilders_smoothing(dx, 14)
    
    # Align indicators to 6h timeframe
    ema_13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_13_1d)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    plus_di_aligned = align_htf_to_ltf(prices, df_1d, plus_di)
    minus_di_aligned = align_htf_to_ltf(prices, df_1d, minus_di)
    
    # Elder Ray Power: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema_13_1d_aligned
    bear_power = low - ema_13_1d_aligned
    
    # Volume confirmation: 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Check for NaN values
        if (np.isnan(ema_13_1d_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(plus_di_aligned[i]) or np.isnan(minus_di_aligned[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        adx_val = adx_aligned[i]
        plus_di_val = plus_di_aligned[i]
        minus_di_val = minus_di_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Regime: Trending (ADX > 25)
            if adx_val > 25:
                # Bullish trend: +DI > -DI
                if plus_di_val > minus_di_val:
                    # Long: Bull Power turns positive with volume
                    if bull_power[i] > 0 and bull_power[i-1] <= 0 and volume_spike[i]:
                        signals[i] = 0.25
                        position = 1
                # Bearish trend: -DI > +DI
                elif minus_di_val > plus_di_val:
                    # Short: Bear Power turns negative with volume
                    if bear_power[i] < 0 and bear_power[i-1] >= 0 and volume_spike[i]:
                        signals[i] = -0.25
                        position = -1
            # Regime: Ranging (ADX < 20)
            elif adx_val < 20:
                # Fade extremes: short at Bull Power peaks, long at Bear Power troughs
                # Short when Bull Power peaks (starts declining from positive)
                if bull_power[i] < bull_power[i-1] and bull_power[i-1] > 0 and volume_spike[i]:
                    signals[i] = -0.25
                    position = -1
                # Long when Bear Power troughs (starts rising from negative)
                elif bear_power[i] > bear_power[i-1] and bear_power[i-1] < 0 and volume_spike[i]:
                    signals[i] = 0.25
                    position = 1
        
        elif position == 1:  # Long position
            # Exit conditions
            exit_signal = False
            # Trending market exit: trend weakens or power fails
            if adx_val > 25:
                if plus_di_val <= minus_di_val:  # Trend turns bearish
                    exit_signal = True
                elif bull_power[i] < 0:  # Bull Power turns negative
                    exit_signal = True
            # Ranging market exit: power normalizes
            elif adx_val < 20:
                if bull_power[i] >= 0:  # Bull Power returns to zero
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit conditions
            exit_signal = False
            # Trending market exit: trend weakens or power fails
            if adx_val > 25:
                if minus_di_val <= plus_di_val:  # Trend turns bullish
                    exit_signal = True
                elif bear_power[i] > 0:  # Bear Power turns positive
                    exit_signal = True
            # Ranging market exit: power normalizes
            elif adx_val < 20:
                if bear_power[i] <= 0:  # Bear Power returns to zero
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals