#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian channel breakout with 1d volume confirmation and ADX trend filter
# - Primary: 4h price breaks above/below 20-period Donchian channel
# - HTF: 1d volume > 1.5x 20-period MA for institutional participation confirmation
# - Trend filter: 4h ADX(14) > 25 to ensure trending market (avoids chop/whipsaws)
# - Long: Close > Upper Donchian + volume confirmation + ADX > 25
# - Short: Close < Lower Donchian + volume confirmation + ADX > 25
# - Exit: Close crosses back inside Donchian channel OR ADX drops below 20 (trend weakening)
# - Position sizing: 0.30 (discrete level, balances return/drawdown)
# - Works in bull/bear: Donchian adapts to volatility, volume filters false breakouts, ADX avoids ranging markets
# - Target: 80-150 total trades over 4 years (20-38/year) for 4h timeframe

name = "4h_1d_donchian_volume_adx_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:  # Need enough data for calculations
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 25:  # Need enough data for calculations
        return np.zeros(n)
    
    # Pre-compute 4h data
    close_4h = prices['close'].values
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    
    # Pre-compute 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 4h Donchian Channel (20-period)
    upper_donchian = np.full(len(close_4h), np.nan)
    lower_donchian = np.full(len(close_4h), np.nan)
    
    for i in range(19, len(close_4h)):
        if not (np.isnan(high_4h[i-19:i+1]).any() or np.isnan(low_4h[i-19:i+1]).any()):
            upper_donchian[i] = np.max(high_4h[i-19:i+1])
            lower_donchian[i] = np.min(low_4h[i-19:i+1])
    
    # Calculate 4h ADX (14-period) for trend strength
    # True Range
    tr = np.full(len(close_4h), np.nan)
    for i in range(1, len(close_4h)):
        if not (np.isnan(high_4h[i]) or np.isnan(low_4h[i]) or np.isnan(close_4h[i-1])):
            tr[i] = max(
                high_4h[i] - low_4h[i],
                abs(high_4h[i] - close_4h[i-1]),
                abs(low_4h[i] - close_4h[i-1])
            )
    
    # Directional Movement
    dm_plus = np.full(len(close_4h), np.nan)
    dm_minus = np.full(len(close_4h), np.nan)
    for i in range(1, len(close_4h)):
        if not (np.isnan(high_4h[i]) or np.isnan(high_4h[i-1]) or 
                np.isnan(low_4h[i]) or np.isnan(low_4h[i-1])):
            up_move = high_4h[i] - high_4h[i-1]
            down_move = low_4h[i-1] - low_4h[i]
            if up_move > down_move and up_move > 0:
                dm_plus[i] = up_move
            elif down_move > up_move and down_move > 0:
                dm_minus[i] = down_move
    
    # Smoothed TR, DM+ and DM- (Wilder's smoothing = EMA with alpha=1/period)
    def wilders_smoothing(values, period):
        result = np.full_like(values, np.nan)
        if len(values) < period:
            return result
        # First value is simple average
        result[period-1] = np.nansum(values[0:period]) / period
        # Subsequent values: Wilder's smoothing
        for i in range(period, len(values)):
            if not np.isnan(values[i]) and not np.isnan(result[i-1]):
                result[i] = (result[i-1] * (period-1) + values[i]) / period
        return result
    
    atr = wilders_smoothing(tr, 14)
    dm_plus_smooth = wilders_smoothing(dm_plus, 14)
    dm_minus_smooth = wilders_smoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.full(len(close_4h), np.nan)
    di_minus = np.full(len(close_4h), np.nan)
    for i in range(len(close_4h)):
        if not (np.isnan(atr[i]) or atr[i] == 0):
            di_plus[i] = (dm_plus_smooth[i] / atr[i]) * 100
            di_minus[i] = (dm_minus_smooth[i] / atr[i]) * 100
    
    # DX and ADX
    dx = np.full(len(close_4h), np.nan)
    for i in range(len(close_4h)):
        if not (np.isnan(di_plus[i]) or np.isnan(di_minus[i]) or 
                (di_plus[i] + di_minus[i]) == 0):
            dx[i] = (abs(di_plus[i] - di_minus[i]) / (di_plus[i] + di_minus[i])) * 100
    
    adx = wilders_smoothing(dx, 14)
    
    # Calculate 1d volume moving average (20-period)
    volume_ma_20_1d = np.full(len(volume_1d), np.nan)
    for i in range(19, len(volume_1d)):
        if not np.isnan(volume_1d[i-19:i+1]).any():
            volume_ma_20_1d[i] = np.mean(volume_1d[i-19:i+1])
    
    # Align all HTF/LTF indicators to 4h timeframe
    upper_donchian_aligned = align_htf_to_ltf(prices, prices, upper_donchian)
    lower_donchian_aligned = align_htf_to_ltf(prices, prices, lower_donchian)
    adx_aligned = align_htf_to_ltf(prices, prices, adx)
    volume_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(upper_donchian_aligned[i]) or np.isnan(lower_donchian_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(volume_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.5x 20-period MA
        volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
        volume_confirm = volume_1d_aligned[i] > 1.5 * volume_ma_20_1d_aligned[i]
        
        # Trend filter: ADX > 25 = strong trend, ADX < 20 = weak trend/ranging
        strong_trend = adx_aligned[i] > 25
        weak_trend = adx_aligned[i] < 20
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Close > Upper Donchian + volume confirmation + strong trend
            if close_4h[i] > upper_donchian_aligned[i] and volume_confirm and strong_trend:
                position = 1
                signals[i] = 0.30
            # Short entry: Close < Lower Donchian + volume confirmation + strong trend
            elif close_4h[i] < lower_donchian_aligned[i] and volume_confirm and strong_trend:
                position = -1
                signals[i] = -0.30
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit: Close crosses back inside Donchian channel OR trend weakens (ADX < 20)
            if position == 1:  # Long position
                if close_4h[i] < lower_donchian_aligned[i] or weak_trend:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.30
            else:  # position == -1 (Short position)
                if close_4h[i] > upper_donchian_aligned[i] or weak_trend:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.30
    
    return signals