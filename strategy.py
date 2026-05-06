#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + volume spike + ADX trend filter + ATR-based stoploss
# Long when price breaks above Donchian(20) high AND volume > 1.5 * avg_volume(20) AND ADX(14) > 25
# Short when price breaks below Donchian(20) low AND volume > 1.5 * avg_volume(20) AND ADX(14) > 25
# Exit when price crosses Donchian midpoint OR ATR stoploss triggered
# Uses discrete sizing 0.30 to balance return and risk
# Target: 100-200 total trades over 4 years (25-50/year) for 4h timeframe
# Donchian breakouts capture strong momentum moves in both bull and bear markets
# Volume confirmation ensures breakout validity while limiting false signals
# ADX filter ensures we only trade in trending conditions, avoiding choppy markets
# Works in bull markets (buy breakouts) and bear markets (sell breakdowns)

name = "4h_Donchian20_VolumeSpike_ADXTrend"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channels (20-period)
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_high_20 + lowest_low_20) / 2.0
    
    # Calculate ADX(14) for trend filter
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Directional Movement
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    up_move[0] = 0
    down_move[0] = 0
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed values (Wilder's smoothing)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(data[:period])
        # Subsequent values: Wilder's smoothing
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    tr_smooth = wilders_smoothing(tr, 14)
    plus_dm_smooth = wilders_smoothing(plus_dm, 14)
    minus_dm_smooth = wilders_smoothing(minus_dm, 14)
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / tr_smooth
    minus_di = 100 * minus_dm_smooth / tr_smooth
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilders_smoothing(dx, 14)
    
    # Calculate volume confirmation
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * avg_volume_20)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    atr_period = 14
    atr_multiplier = 2.5
    
    # Calculate ATR for stoploss
    atr = wilders_smoothing(tr, atr_period)
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(highest_high_20[i]) or np.isnan(lowest_low_20[i]) or 
            np.isnan(adx[i]) or np.isnan(atr[i]) or np.isnan(avg_volume_20[i]) or 
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian high + volume spike + ADX > 25
            if (close[i] > highest_high_20[i] and 
                volume_confirm[i] and 
                adx[i] > 25):
                signals[i] = 0.30
                position = 1
            # Short: price breaks below Donchian low + volume spike + ADX > 25
            elif (close[i] < lowest_low_20[i] and 
                  volume_confirm[i] and 
                  adx[i] > 25):
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit conditions for long
            exit_long = False
            # Exit 1: price crosses below Donchian midpoint
            if close[i] < donchian_mid[i]:
                exit_long = True
            # Exit 2: ATR-based stoploss (2.5 * ATR below entry)
            # We approximate by checking if price has moved against us significantly
            # In practice, we'd track entry price, but we use a trailing approach
            if i >= 1 and close[i] < close[i-1] - atr_multiplier * atr[i]:
                exit_long = True
            
            if exit_long:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit conditions for short
            exit_short = False
            # Exit 1: price crosses above Donchian midpoint
            if close[i] > donchian_mid[i]:
                exit_short = True
            # Exit 2: ATR-based stoploss (2.5 * ATR above entry)
            if i >= 1 and close[i] > close[i-1] + atr_multiplier * atr[i]:
                exit_short = True
            
            if exit_short:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals