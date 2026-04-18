#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume confirmation and ADX trend filter
# - Long: break above Donchian(20) high + volume > 1.5x 20-period avg + ADX > 20
# - Short: break below Donchian(20) low + volume > 1.5x 20-period avg + ADX > 20
# - Exit: opposite Donchian break or ADX < 15 (trend weakening)
# - Uses 1d ATR for volatility filter (avoid choppy markets)
# Target: 20-50 trades/year to minimize fee drag while capturing major moves
name = "4h_Donchian20_Volume_ADXTrend"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ATR filter (trading only when volatility is sufficient)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Donchian channels (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate ADX (14-period) for trend strength
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
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values (using Wilder's smoothing = EMA with alpha=1/period)
    def wilders_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            # First value is simple average
            result[period-1] = np.mean(data[:period])
            # Subsequent values: Wilder's smoothing
            for i in range(period, len(data)):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr = wilders_smooth(tr, 14)
    plus_di = 100 * wilders_smooth(plus_dm, 14) / atr
    minus_di = 100 * wilders_smooth(minus_dm, 14) / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilders_smooth(dx, 14)
    
    # Calculate 1d ATR for volatility filter (only trade when volatility is sufficient)
    high_1d = pd.Series(df_1d['high'].values)
    low_1d = pd.Series(df_1d['low'].values)
    close_1d = pd.Series(df_1d['close'].values)
    
    tr1_1d = high_1d - low_1d
    tr2_1d = np.abs(high_1d - np.roll(close_1d, 1))
    tr3_1d = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))
    tr_1d[0] = tr1_1d[0]
    
    def wilders_smooth_1d(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            result[period-1] = np.mean(data[:period])
            for i in range(period, len(data)):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr_1d = wilders_smooth_1d(tr_1d, 14)
    atr_1d_avg = pd.Series(atr_1d).rolling(window=10, min_periods=10).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d_avg)
    
    # Current ATR ratio (vs 10-period average) to detect low volatility
    atr_current = atr
    atr_ma_10 = pd.Series(atr_current).rolling(window=10, min_periods=10).mean().values
    vol_filter = atr_current > (0.8 * atr_ma_10)  # Avoid extremely low volatility
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or 
            np.isnan(adx[i]) or np.isnan(atr_1d_aligned[i]) or
            np.isnan(vol_ma_20[i]) or np.isnan(atr_ma_10[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: ADX > 20 indicates strong trend
        strong_trend = adx[i] > 20
        weak_trend = adx[i] < 15  # Exit when trend weakens
        
        if position == 0:
            # Long: break above Donchian high + volume confirm + strong trend + sufficient volatility
            if (close[i] > high_roll[i] and 
                volume_confirm[i] and 
                strong_trend and 
                vol_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian low + volume confirm + strong trend + sufficient volatility
            elif (close[i] < low_roll[i] and 
                  volume_confirm[i] and 
                  strong_trend and 
                  vol_filter[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: break below Donchian low OR trend weakens
            if close[i] < low_roll[i] or weak_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: break above Donchian high OR trend weakens
            if close[i] > high_roll[i] or weak_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals