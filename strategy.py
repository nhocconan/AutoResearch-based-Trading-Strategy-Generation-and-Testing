# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour trend following with 12-hour ADX filter and volume confirmation
# Uses ADX(14) from 12h timeframe to filter only strong trends (ADX > 25)
# Enters on 6h EMA(21) cross of EMA(55) with volume > 1.5x 20-period average
# Exits on opposite EMA cross
# Designed to work in both bull and bear markets by only trading strong trends
# Targets 50-150 total trades over 4 years (12-37/year) to minimize fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 6h EMA(21) and EMA(55)
    ema_21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_55 = pd.Series(close).ewm(span=55, adjust=False, min_periods=55).mean().values
    
    # Calculate 12h ADX(14) for trend strength filter
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = high_12h[1:] - low_12h[1:]
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    # Directional Movement
    dm_plus = np.where((high_12h[1:] - high_12h[:-1]) > (low_12h[:-1] - low_12h[1:]), 
                       np.maximum(high_12h[1:] - high_12h[:-1], 0), 0)
    dm_minus = np.where((low_12h[:-1] - low_12h[1:]) > (high_12h[1:] - high_12h[:-1]), 
                        np.maximum(low_12h[:-1] - low_12h[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smooth TR, DM+, DM- with Wilder's smoothing (alpha = 1/period)
    def wilder_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            # First value is simple average
            result[period-1] = np.nanmean(data[1:period])
            # Subsequent values: Wilder smoothing
            for i in range(period, len(data)):
                if not np.isnan(result[i-1]) and not np.isnan(data[i]):
                    result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr = wilder_smooth(tr, 14)
    dm_plus_smooth = wilder_smooth(dm_plus, 14)
    dm_minus_smooth = wilder_smooth(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr != 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr != 0, 100 * dm_minus_smooth / atr, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = wilder_smooth(dx, 14)
    
    # Calculate 12h volume average (20-period)
    vol_12h = df_12h['volume'].values
    vol_ma_12h = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
    
    # Align indicators to 6h timeframe
    ema_21_aligned = ema_21  # Already on 6h
    ema_55_aligned = ema_55  # Already on 6h
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 60  # for EMA(55) and ADX calculations
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_21_aligned[i]) or np.isnan(ema_55_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: EMA21 crosses above EMA55 with strong trend (ADX > 25) and volume confirmation
            if (ema_21_aligned[i] > ema_55_aligned[i] and 
                ema_21_aligned[i-1] <= ema_55_aligned[i-1] and  # Cross just happened
                adx_aligned[i] > 25 and
                volume[i] > 1.5 * vol_ma_12h_aligned[i]):
                position = 1
                signals[i] = position_size
            # Short: EMA21 crosses below EMA55 with strong trend (ADX > 25) and volume confirmation
            elif (ema_21_aligned[i] < ema_55_aligned[i] and 
                  ema_21_aligned[i-1] >= ema_55_aligned[i-1] and  # Cross just happened
                  adx_aligned[i] > 25 and
                  volume[i] > 1.5 * vol_ma_12h_aligned[i]):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: EMA21 crosses below EMA55
            if ema_21_aligned[i] < ema_55_aligned[i] and ema_21_aligned[i-1] >= ema_55_aligned[i-1]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: EMA21 crosses above EMA55
            if ema_21_aligned[i] > ema_55_aligned[i] and ema_21_aligned[i-1] <= ema_55_aligned[i-1]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_ADX_EMA_Cross_Volume"
timeframe = "6h"
leverage = 1.0