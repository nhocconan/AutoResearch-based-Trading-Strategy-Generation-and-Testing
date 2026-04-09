#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d trend filter (ADX>25) and volume confirmation
# - Uses 1d HTF for trend regime (ADX>25 = trending, <20 = ranging)
# - 4h Donchian channel (20-period high/low) for breakout entries
# - Long on break above upper band in uptrend, short on break below lower band in downtrend
# - Volume confirmation: current 4h volume > 1.5x 20-period average
# - ATR-based trailing stop: exit long when price < highest_high - 2*ATR, exit short when price < lowest_low + 2*ATR
# - Fixed position size 0.25 to control drawdown
# - Target: 20-50 trades/year on 4h timeframe (80-200 total over 4 years)

name = "4h_1d_donchian_breakout_trend_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d ADX for trend regime filter
    # ADX calculation: +DI, -DI, DX, then smoothed ADX
    period = 14
    up_move = np.diff(high_1d, prepend=high_1d[0])
    down_move = -np.diff(low_1d, prepend=low_1d[0])
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    tr = np.maximum(np.abs(np.diff(close_1d, prepend=close_1d[0])),
                    np.maximum(high_1d - low_1d,
                               np.maximum(np.abs(high_1d - np.roll(close_1d, 1)),
                                          np.absolute(np.roll(high_1d, 1) - low_1d))))
    tr[0] = high_1d[0] - low_1d[0]
    
    # Smooth TR, +DM, -DM using Wilder's smoothing (alpha = 1/period)
    def wilder_smooth(data, period):
        result = np.full_like(data, np.nan)
        alpha = 1.0 / period
        result[period-1] = np.nansum(data[:period])
        for i in range(period, len(data)):
            result[i] = result[i-1] * (1 - alpha) + data[i] * alpha
        return result
    
    atr_1d = wilder_smooth(tr, period)
    plus_di_1d = 100 * wilder_smooth(plus_dm, period) / atr_1d
    minus_di_1d = 100 * wilder_smooth(minus_dm, period) / atr_1d
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)
    adx_1d = wilder_smooth(dx_1d, period)
    
    # Align 1d ADX to 4h timeframe (wait for completed 1d bar)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate 4h Donchian channel (20-period)
    donchian_period = 20
    upper_band = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    lower_band = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # Calculate 4h ATR for trailing stop
    atr_period = 14
    tr_4h = np.maximum(np.abs(np.diff(close, prepend=close[0])),
                       np.maximum(high - low,
                                  np.maximum(np.abs(high - np.roll(close, 1)),
                                             np.abs(np.roll(high, 1) - low))))
    tr_4h[0] = high[0] - low[0]
    atr_4h = pd.Series(tr_4h).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    # Pre-compute volume confirmation (20-period average for 4h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    highest_high = 0.0  # for long trailing stop
    lowest_low = 0.0    # for short trailing stop
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or
            np.isnan(atr_4h[i]) or np.isnan(vol_ma_20[i]) or vol_ma_20[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4h volume > 1.5x average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        # Trend regime filter: ADX > 25 = trending (trade breakouts), ADX < 20 = ranging (avoid)
        trending = adx_1d_aligned[i] > 25.0
        ranging = adx_1d_aligned[i] < 20.0
        
        # Fixed position size
        position_size = 0.25
        
        if position == 1:  # Long position
            # Update highest high for trailing stop
            highest_high = max(highest_high, high[i])
            # Exit when price closes below highest_high - 2*ATR (trailing stop)
            if close[i] < highest_high - 2.0 * atr_4h[i]:
                position = 0
                highest_high = 0.0
                signals[i] = 0.0
            else:
                signals[i] = position_size
                
        elif position == -1:  # Short position
            # Update lowest low for trailing stop
            lowest_low = min(lowest_low, low[i])
            # Exit when price closes above lowest_low + 2*ATR (trailing stop)
            if close[i] > lowest_low + 2.0 * atr_4h[i]:
                position = 0
                lowest_low = 0.0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
        else:  # Flat
            # Breakout entry with volume confirmation and trend regime filter
            if volume_confirmed and trending:
                # Long: break above upper band
                if close[i] > upper_band[i]:
                    position = 1
                    highest_high = high[i]
                    signals[i] = position_size
                # Short: break below lower band
                elif close[i] < lower_band[i]:
                    position = -1
                    lowest_low = low[i]
                    signals[i] = -position_size
    
    return signals