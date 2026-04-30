#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d ADX regime filter and volume confirmation.
# Bull Power = High - EMA13, Bear Power = EMA13 - Low.
# Long when Bull Power > 0, Bear Power < 0, 1d ADX > 25 (trending), and volume > 1.5x 20-bar avg.
# Short when Bear Power > 0, Bull Power < 0, 1d ADX > 25, and volume > 1.5x 20-bar avg.
# Exit when Elder Power diverges (Bull Power <= 0 for long, Bear Power <= 0 for short) or ADX < 20.
# Uses 1d ADX to filter for trending markets only, avoiding whipsaws in ranges.
# Elder Ray captures the underlying bull/bear strength behind price moves.
# Volume confirmation ensures breakouts have participation.
# Discrete position sizing at ±0.25 to balance performance and fee drag.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.
# Works in bull markets via strong Bull Power and in bear markets via strong Bear Power.

name = "6h_ElderRay_1dADX_Trend_VolumeConfirm_v1"
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
    
    # Load 1d data ONCE before loop for ADX calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ADX (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed TR, DM+, DM- (Wilder's smoothing = EMA with alpha=1/period)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(data[:period])
        # Rest is Wilder's smoothing
        alpha = 1.0 / period
        for i in range(period, len(data)):
            if not np.isnan(data[i]):
                result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
            else:
                result[i] = result[i-1]
        return result
    
    tr_smoothed = wilders_smoothing(tr, 14)
    dm_plus_smoothed = wilders_smoothing(dm_plus, 14)
    dm_minus_smoothed = wilders_smoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(tr_smoothed != 0, 100 * dm_plus_smoothed / tr_smoothed, 0)
    di_minus = np.where(tr_smoothed != 0, 100 * dm_minus_smoothed / tr_smoothed, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = wilders_smoothing(dx, 14)
    
    # Align 1d ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate Elder Ray (Bull/Bear Power) on 6h timeframe
    # Bull Power = High - EMA13(Close)
    # Bear Power = EMA13(Close) - Low
    close_series = pd.Series(close)
    ema_13 = close_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13
    bear_power = ema_13 - low
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # warmup for EMA13 and ADX
    
    for i in range(start_idx, n):
        # Skip if indicators not available
        if (np.isnan(ema_13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        curr_bull_power = bull_power[i]
        curr_bear_power = bear_power[i]
        curr_adx = adx_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Bull Power > 0, Bear Power < 0, ADX > 25 (trending up), volume spike
            if (curr_bull_power > 0 and 
                curr_bear_power < 0 and 
                curr_adx > 25 and 
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: Bear Power > 0, Bull Power < 0, ADX > 25 (trending down), volume spike
            elif (curr_bear_power > 0 and 
                  curr_bull_power < 0 and 
                  curr_adx > 25 and 
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position
            # Exit conditions: Bull Power <= 0 (loss of bullish momentum) OR ADX < 20 (trend weakening)
            if curr_bull_power <= 0 or curr_adx < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit conditions: Bear Power <= 0 (loss of bearish momentum) OR ADX < 20 (trend weakening)
            if curr_bear_power <= 0 or curr_adx < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals