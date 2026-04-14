#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d ADX Trend Filter and Volume Spike
# Elder Ray measures bull/bear power via EMA(13) to gauge trend strength
# 1d ADX (>25) confirms trending conditions to avoid whipsaws in ranging markets
# Volume spike (>2x average) ensures institutional participation
# Works in bull markets via bull power strength and in bear markets via bear power strength
# Target: 15-35 trades/year (60-140 total over 4 years) to minimize fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d ADX data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    # Calculate ADX(14) on 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
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
    
    # Smooth TR, DM+, DM- with Wilder's smoothing (alpha = 1/period)
    def wildeR_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            result[period-1] = np.nansum(data[:period])
            for i in range(period, len(data)):
                result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    atr = wildeR_smooth(tr, 14)
    dm_plus_smooth = wildeR_smooth(dm_plus, 14)
    dm_minus_smooth = wildeR_smooth(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr != 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr != 0, 100 * dm_minus_smooth / atr, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = wildeR_smooth(dx, 14)
    
    # Align ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Elder Ray: Bull Power = High - EMA(13), Bear Power = EMA(13) - Low
    close_series = pd.Series(close)
    ema_13 = close_series.ewm(span=13, adjust=False).values
    
    bull_power = high - ema_13
    bear_power = ema_13 - low
    
    # Volume confirmation: volume > 2x average volume (20-period)
    vol_series = pd.Series(volume)
    avg_vol = vol_series.rolling(window=20, min_periods=20).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 35  # for EMA(13) and ADX calculations
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_13[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(avg_vol[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: only trade when ADX > 25 (trending market)
        trending = adx_aligned[i] > 25
        
        if position == 0:
            # Long: Bull Power > 0 and increasing with volume filter in uptrend
            if bull_power[i] > 0 and bull_power[i] > bull_power[i-1] and trending and volume[i] > 2.0 * avg_vol[i]:
                position = 1
                signals[i] = position_size
            # Short: Bear Power > 0 and increasing with volume filter in downtrend
            elif bear_power[i] > 0 and bear_power[i] > bear_power[i-1] and trending and volume[i] > 2.0 * avg_vol[i]:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Bull Power becomes negative or turns down
            if bull_power[i] <= 0 or bull_power[i] < bull_power[i-1]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Bear Power becomes negative or turns down
            if bear_power[i] <= 0 or bear_power[i] < bear_power[i-1]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_ElderRay_1dADX_Volume"
timeframe = "6h"
leverage = 1.0