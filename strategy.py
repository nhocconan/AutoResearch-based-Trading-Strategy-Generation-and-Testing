#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 4h Daily Camarilla Pivot with Volume and ADX Filter
# Hypothesis: Camarilla pivot levels from daily timeframe provide strong intraday
# support/resistance. Breakouts above R4 or below S4 with volume and daily ADX > 25
# capture momentum moves. Daily trend filter reduces whipsaws in both bull and bear markets.
# Target: 25-40 trades/year (100-160 total over 4 years) to minimize fee drag.

name = "4h_daily_camarilla_pivot_volume_adx_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivots and ADX
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 30:
        return np.zeros(n)
    
    # Calculate ADX on daily data
    daily_high = df_daily['high'].values
    daily_low = df_daily['low'].values
    daily_close = df_daily['close'].values
    
    # True Range
    tr1 = daily_high[1:] - daily_low[1:]
    tr2 = np.abs(daily_high[1:] - daily_close[:-1])
    tr3 = np.abs(daily_low[1:] - daily_close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((daily_high[1:] - daily_high[:-1]) > (daily_low[:-1] - daily_low[1:]),
                       np.maximum(daily_high[1:] - daily_high[:-1], 0), 0)
    dm_minus = np.where((daily_low[:-1] - daily_low[1:]) > (daily_high[1:] - daily_high[:-1]),
                        np.maximum(daily_low[:-1] - daily_low[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Wilder's smoothing function
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan, dtype=float)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nansum(data[1:period]) / period
        # Subsequent values
        for i in range(period, len(data)):
            if not np.isnan(result[i-1]):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr = wilders_smoothing(tr, 14)
    dm_plus_smooth = wilders_smoothing(dm_plus, 14)
    dm_minus_smooth = wilders_smoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr > 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr > 0, 100 * dm_minus_smooth / atr, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) > 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = wilders_smoothing(dx, 14)
    
    # Daily Camarilla pivot levels (based on previous day's range)
    # R4 = Close + 1.5 * (High - Low)
    # R3 = Close + 1.25 * (High - Low)
    # R2 = Close + 1.0 * (High - Low)
    # R1 = Close + 0.75 * (High - Low)
    # PP = (High + Low + Close) / 3
    # S1 = Close - 0.75 * (High - Low)
    # S2 = Close - 1.0 * (High - Low)
    # S3 = Close - 1.25 * (High - Low)
    # S4 = Close - 1.5 * (High - Low)
    prev_high = np.concatenate([[np.nan], daily_high[:-1]])
    prev_low = np.concatenate([[np.nan], daily_low[:-1]])
    prev_close = np.concatenate([[np.nan], daily_close[:-1]])
    
    camarilla_high = prev_high - prev_low
    camarilla_r4 = prev_close + 1.5 * camarilla_high
    camarilla_s4 = prev_close - 1.5 * camarilla_high
    
    # Align daily indicators to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_daily, adx)
    r4_aligned = align_htf_to_ltf(prices, df_daily, camarilla_r4)
    s4_aligned = align_htf_to_ltf(prices, df_daily, camarilla_s4)
    
    # Volume filter on 4h: volume > 1.5x 30-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=30, min_periods=30).mean().values
    vol_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):
        # Skip if required data not available
        if (np.isnan(adx_aligned[i]) or np.isnan(r4_aligned[i]) or
            np.isnan(s4_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price falls back below S4 or ADX weakens
            if close[i] < s4_aligned[i] or adx_aligned[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: price rises back above R4 or ADX weakens
            if close[i] > r4_aligned[i] or adx_aligned[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Strong trend required
            if adx_aligned[i] >= 25:
                # Long entry: breakout above R4 with volume
                if (high[i] > r4_aligned[i] and close[i] > r4_aligned[i] and
                    vol_filter[i]):
                    position = 1
                    signals[i] = 0.25
                # Short entry: breakdown below S4 with volume
                elif (low[i] < s4_aligned[i] and close[i] < s4_aligned[i] and
                      vol_filter[i]):
                    position = -1
                    signals[i] = -0.25
    
    return signals