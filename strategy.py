#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using weekly Donchian channel breakout with volume confirmation and ADX filter.
# Long when price breaks above weekly Donchian high (20-week period) with ADX > 25 and volume > 1.5x average.
# Short when price breaks below weekly Donchian low with same conditions.
# Exit when price crosses the weekly Donchian midline.
# Targets 7-25 trades per year to avoid fee drag, suitable for both bull and bear markets via trend filter.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Donchian channels and ADX
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 30:
        return np.zeros(n)
    
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    close_weekly = df_weekly['close'].values
    volume_weekly = df_weekly['volume'].values
    
    # Calculate weekly ADX (14-period)
    def calculate_adx(high, low, close, period=14):
        n = len(high)
        if n < period * 2:
            return np.full(n, np.nan)
        
        # True Range
        tr = np.maximum(high - low, 
                        np.maximum(np.abs(high - np.concatenate([[np.nan], close[:-1]])),
                                   np.abs(np.concatenate([[np.nan], low[:-1]]) - low)))
        
        # Directional Movement
        dm_plus = np.where((high - np.concatenate([[np.nan], high[:-1]])) > 
                           (np.concatenate([[np.nan], low[:-1]]) - low), 
                           np.maximum(high - np.concatenate([[np.nan], high[:-1]]), 0), 0)
        dm_minus = np.where((np.concatenate([[np.nan], low[:-1]]) - low) > 
                            (high - np.concatenate([[np.nan], high[:-1]])), 
                            np.maximum(np.concatenate([[np.nan], low[:-1]]) - low, 0), 0)
        
        # Smooth TR, DM+, DM- using Wilder's smoothing (alpha = 1/period)
        def wilder_smooth(data, period):
            n = len(data)
            result = np.full(n, np.nan)
            if n < period:
                return result
            # First value: simple average
            result[period-1] = np.nansum(data[:period])
            # Subsequent values: Wilder smoothing
            for i in range(period, n):
                result[i] = result[i-1] - (result[i-1] / period) + data[i]
            return result
        
        tr_smooth = wilder_smooth(tr, period)
        dm_plus_smooth = wilder_smooth(dm_plus, period)
        dm_minus_smooth = wilder_smooth(dm_minus, period)
        
        # Directional Indicators
        plus_di = 100 * dm_plus_smooth / tr_smooth
        minus_di = 100 * dm_minus_smooth / tr_smooth
        
        # DX and ADX
        dx = np.where((plus_di + minus_di) != 0, 
                      100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
        adx = wilder_smooth(dx, period)
        
        return adx
    
    adx_weekly = calculate_adx(high_weekly, low_weekly, close_weekly, 14)
    
    # Calculate weekly Donchian channels (20-period)
    donch_period = 20
    donch_high_weekly = np.full(len(high_weekly), np.nan)
    donch_low_weekly = np.full(len(low_weekly), np.nan)
    donch_mid_weekly = np.full(len(close_weekly), np.nan)
    
    for i in range(donch_period - 1, len(high_weekly)):
        donch_high_weekly[i] = np.max(high_weekly[i - donch_period + 1:i + 1])
        donch_low_weekly[i] = np.min(low_weekly[i - donch_period + 1:i + 1])
        donch_mid_weekly[i] = (donch_high_weekly[i] + donch_low_weekly[i]) / 2
    
    # Align weekly indicators to daily timeframe
    donch_high_aligned = align_htf_to_ltf(prices, df_weekly, donch_high_weekly)
    donch_low_aligned = align_htf_to_ltf(prices, df_weekly, donch_low_weekly)
    donch_mid_aligned = align_htf_to_ltf(prices, df_weekly, donch_mid_weekly)
    adx_aligned = align_htf_to_ltf(prices, df_weekly, adx_weekly)
    
    # Volume MA for confirmation (20-period)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i - 19:i + 1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Donchian channels, ADX, and volume MA20
    start_idx = max(donch_period - 1, 14*2, 19)  # ADX needs ~28 bars for stability
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or 
            np.isnan(donch_mid_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Volume filter: require volume above average
        vol_filter = vol_now > 1.5 * vol_avg
        # Trend filter: require ADX > 25
        trend_filter = adx_aligned[i] > 25
        
        if position == 0:
            # Long: break above weekly Donchian high with ADX > 25 and volume filter
            if (price > donch_high_aligned[i] and 
                trend_filter and vol_filter):
                signals[i] = size
                position = 1
            # Short: break below weekly Donchian low with ADX > 25 and volume filter
            elif (price < donch_low_aligned[i] and 
                  trend_filter and vol_filter):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below weekly Donchian midline
            if price < donch_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above weekly Donchian midline
            if price > donch_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_WeeklyDonchian20_ADX25_Volume"
timeframe = "1d"
leverage = 1.0