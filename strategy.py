# %%
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1-day Donchian(20) breakout with volume confirmation and ADX trend filter
# Long when price breaks above 1-day Donchian upper channel with volume > 1.5x average and ADX > 25
# Short when price breaks below 1-day Donchian lower channel with volume > 1.5x average and ADX > 25
# Donchian channels capture momentum breakouts, volume confirms strength, ADX ensures trending market
# Works in bull/bear markets: breakouts capture momentum, ADX filter avoids range-bound false signals
# Target: 15-30 trades per year (60-120 over 4 years) with 0.25 position sizing

name = "12h_1dDonchian20_Volume_ADX_Breakout_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1-day Donchian(20) channels ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 20-period high and low for Donchian channels
    high_20 = df_1d['high'].rolling(window=20, min_periods=20).max().values
    low_20 = df_1d['low'].rolling(window=20, min_periods=20).min().values
    
    # Align 1-day Donchian levels to 12h timeframe
    upper_20_aligned = align_htf_to_ltf(prices, df_1d, high_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_1d, low_20)
    
    # 1-day ADX(14) for trend filter
    # Calculate True Range
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
    
    # Calculate Directional Movement
    up_move = df_1d['high'] - df_1d['high'].shift(1)
    down_move = df_1d['low'].shift(1) - df_1d['low']
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smooth TR and DM using Wilder's smoothing (alpha = 1/period)
    def wilder_smooth(series, period):
        result = np.full_like(series, np.nan, dtype=float)
        if len(series) < period:
            return result
        # First value is simple average
        result[period-1] = np.nansum(series[:period])
        # Subsequent values: smoothed = previous_smoothed - (previous_smoothed/period) + current
        for i in range(period, len(series)):
            result[i] = result[i-1] - (result[i-1]/period) + series[i]
        return result
    
    atr = wilder_smooth(tr.values, 14)
    plus_di = 100 * wilder_smooth(plus_dm, 14) / atr
    minus_di = 100 * wilder_smooth(minus_dm, 14) / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilder_smooth(dx, 14)
    
    # Align 1-day ADX to 12h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume confirmation: >1.5x 50-period average
    vol_ma_50 = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    volume_filter = volume > (1.5 * vol_ma_50)
    
    # Pre-compute session filter (00-24 UTC - trade all hours for 12h)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = np.ones(n, dtype=bool)  # Trade all hours for 12h timeframe
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any critical value is NaN
        if (np.isnan(upper_20_aligned[i]) or np.isnan(lower_20_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price breaks above upper Donchian with volume and trend confirmation
            if close[i] > upper_20_aligned[i] and volume_filter[i] and adx_aligned[i] > 25:
                signals[i] = 0.25
                position = 1
            # Short breakout: price breaks below lower Donchian with volume and trend confirmation
            elif close[i] < lower_20_aligned[i] and volume_filter[i] and adx_aligned[i] > 25:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below lower Donchian (failed support) or trend weakens
            if close[i] < lower_20_aligned[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above upper Donchian (failed resistance) or trend weakens
            if close[i] > upper_20_aligned[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# %%