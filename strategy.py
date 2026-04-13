#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Donchian breakout with 12h/1d regime filter and volume confirmation
    # Long: price breaks above Donchian(20) high AND 12h ADX > 25 (trending) AND volume > 1.5x 20-period average
    # Short: price breaks below Donchian(20) low AND 12h ADX > 25 AND volume > 1.5x 20-period average
    # Exit: price returns to Donchian midpoint OR ADX < 20 (range) OR volume dry-up
    # Using 6h for price action, 12h for regime (ADX), volume confirmation on 6h
    # Session filter (08-20 UTC) to reduce noise trades
    # Discrete position sizing (0.25) to balance return and drawdown
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 6h data for Donchian channels (call ONCE before loop)
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    
    # Donchian channels (20-period)
    donch_high_6h = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    donch_low_6h = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
    donch_mid_6h = (donch_high_6h + donch_low_6h) / 2
    
    # Align 6h Donchian levels to 6h (wait for completed 6h bar)
    donch_high_6h_aligned = align_htf_to_ltf(prices, df_6h, donch_high_6h)
    donch_low_6h_aligned = align_htf_to_ltf(prices, df_6h, donch_low_6h)
    donch_mid_6h_aligned = align_htf_to_ltf(prices, df_6h, donch_mid_6h)
    
    # Get 12h data for ADX regime filter (call ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate ADX (14-period) on 12h
    # True Range
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Directional Movement
    dm_plus = np.where((high_12h - np.roll(high_12h, 1)) > (np.roll(low_12h, 1) - low_12h),
                       np.maximum(high_12h - np.roll(high_12h, 1), 0), 0)
    dm_minus = np.where((np.roll(low_12h, 1) - low_12h) > (high_12h - np.roll(high_12h, 1)),
                        np.maximum(np.roll(low_12h, 1) - low_12h, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed TR, DM+ (Wilder's smoothing)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            result[period-1] = np.mean(data[:period])
            for i in range(period, len(data)):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr_12h = wilders_smoothing(tr, 14)
    dm_plus_smooth = wilders_smoothing(dm_plus, 14)
    dm_minus_smooth = wilders_smoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr_12h != 0, dm_plus_smooth / atr_12h * 100, 0)
    di_minus = np.where(atr_12h != 0, dm_minus_smooth / atr_12h * 100, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100, 0)
    adx_12h = wilders_smoothing(dx, 14)
    
    # Align 12h ADX to 6h (wait for completed 12h bar)
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # Volume confirmation: >1.5x 20-period average on 6h
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(donch_high_6h_aligned[i]) or np.isnan(donch_low_6h_aligned[i]) or 
            np.isnan(donch_mid_6h_aligned[i]) or np.isnan(adx_12h_aligned[i]) or 
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 08-20 UTC
        if not in_session[i]:
            # Force flat outside session
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        # Regime filter: only trade when ADX > 25 (trending market)
        regime_ok = adx_12h_aligned[i] > 25
        
        # Volume confirmation
        vol_confirm = volume_spike[i]
        
        # Entry logic: Donchian breakout + regime + volume + session
        long_entry = (close[i] > donch_high_6h_aligned[i]) and regime_ok and vol_confirm
        short_entry = (close[i] < donch_low_6h_aligned[i]) and regime_ok and vol_confirm
        
        # Exit logic: return to midpoint OR regime change OR volume dry-up
        long_exit = (close[i] < donch_mid_6h_aligned[i]) or (adx_12h_aligned[i] < 20) or not vol_confirm
        short_exit = (close[i] > donch_mid_6h_aligned[i]) or (adx_12h_aligned[i] < 20) or not vol_confirm
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_12h_donchian_breakout_adx_volume_v1"
timeframe = "6h"
leverage = 1.0