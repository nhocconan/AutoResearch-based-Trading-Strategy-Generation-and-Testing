#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Donchian channel breakout with volume confirmation and ADX regime filter
# Long when price breaks above 1d Donchian upper channel AND ADX(14) > 25 AND volume > 1.5 * avg_volume(20)
# Short when price breaks below 1d Donchian lower channel AND ADX(14) > 25 AND volume > 1.5 * avg_volume(20)
# Exit when price crosses 1d EMA50 (trend filter reversal)
# Uses discrete sizing 0.30 to balance return and drawdown
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe
# 1d Donchian provides clear structure with proven breakout edge
# ADX > 25 ensures we only trade in trending markets (works in bull/bear)
# Volume confirmation filters weak breakouts (reduces false signals)

name = "4h_1dDonchian20_ADX25_Volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for Donchian channels and EMA50
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need sufficient data for EMA50 and Donchian(20)
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Donchian channels (20-period)
    high_series_1d = pd.Series(high_1d)
    low_series_1d = pd.Series(low_1d)
    donchian_upper_1d = high_series_1d.rolling(window=20, min_periods=20).max().values
    donchian_lower_1d = low_series_1d.rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d EMA50 for exit signal
    close_series_1d = pd.Series(close_1d)
    ema_50_1d = close_series_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 1d ADX(14) for regime filter
    # ADX calculation: +DI, -DI, DX, then smoothed ADX
    # +DM = max(high - previous high, 0) if high - previous high > previous low - low else 0
    # -DM = max(previous low - low, 0) if previous low - low > high - previous high else 0
    # TR = max(high - low, abs(high - previous close), abs(low - previous close))
    # +DI = 100 * smoothed(+DM) / smoothed(TR)
    # -DI = 100 * smoothed(-DM) / smoothed(TR)
    # DX = 100 * abs(+DI - -DI) / (+DI + -DI)
    # ADX = smoothed(DX)
    
    # Calculate True Range (TR)
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr3 = np.abs(low_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Calculate +DM and -DM
    up_move = high_1d - np.concatenate([[high_1d[0]], high_1d[:-1]])
    down_move = np.concatenate([[low_1d[0]], low_1d[:-1]]) - low_1d
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smooth TR, +DM, -DM using Wilder's smoothing (equivalent to EMA with alpha=1/period)
    def wilder_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            # First value is simple average
            result[period-1] = np.mean(data[:period])
            # Subsequent values: prev * (1 - 1/period) + current * (1/period)
            for i in range(period, len(data)):
                result[i] = result[i-1] * (1 - 1/period) + data[i] * (1/period)
        return result
    
    atr_1d = wilder_smooth(tr, 14)
    plus_di_1d = 100 * wilder_smooth(plus_dm, 14) / atr_1d
    minus_di_1d = 100 * wilder_smooth(minus_dm, 14) / atr_1d
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)
    adx_1d = wilder_smooth(dx_1d, 14)
    
    # Align 1d indicators to 4h timeframe (wait for completed 1d bar)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper_1d)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower_1d)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate volume confirmation: volume > 1.5 * 20-period average volume
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(adx_1d_aligned[i]) or 
            np.isnan(avg_volume_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 1d Donchian upper with ADX > 25 and volume confirmation
            if (close[i] > donchian_upper_aligned[i] and close[i-1] <= donchian_upper_aligned[i-1] and 
                adx_1d_aligned[i] > 25 and volume_confirm[i]):
                signals[i] = 0.30
                position = 1
            # Short: price breaks below 1d Donchian lower with ADX > 25 and volume confirmation
            elif (close[i] < donchian_lower_aligned[i] and close[i-1] >= donchian_lower_aligned[i-1] and 
                  adx_1d_aligned[i] > 25 and volume_confirm[i]):
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: price crosses below 1d EMA50 (trend filter reversal)
            if close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: price crosses above 1d EMA50 (trend filter reversal)
            if close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals