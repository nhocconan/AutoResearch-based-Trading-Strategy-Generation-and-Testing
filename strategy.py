#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla Pivot Breakout with daily volume confirmation and weekly ADX trend filter.
# Uses Camarilla pivot levels (H3/L3) from daily timeframe for institutional breakout levels.
# Requires volume spike above 20-period average for confirmation.
# Uses weekly ADX > 25 to ensure trading only in trending markets, avoiding chop.
# Designed for low trade frequency (12-37/year) to minimize fee drag in 12h timeframe.
# Works in bull markets (breakouts above H3) and bear markets (breakouts below L3).
name = "12h_Camarilla_H3L3_Volume_ADXFilter"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivots and volume (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels (H3, L3) using previous day's data
    # H3 = close + 1.1*(high - low)/6
    # L3 = close - 1.1*(high - low)/6
    high_d = df_1d['high'].values
    low_d = df_1d['low'].values
    close_d = df_1d['close'].values
    
    # Calculate pivot components
    hl_range = high_d - low_d
    h3 = close_d + 1.1 * hl_range / 6
    l3 = close_d - 1.1 * hl_range / 6
    
    # Align daily H3/L3 to 12h timeframe (wait for daily close)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    
    # Get weekly data for ADX trend filter (ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly ADX (14-period)
    high_w = df_1w['high'].values
    low_w = df_1w['low'].values
    close_w = df_1w['close'].values
    
    # True Range
    tr1 = high_w[1:] - low_w[1:]
    tr2 = np.abs(high_w[1:] - close_w[:-1])
    tr3 = np.abs(low_w[1:] - close_w[:-1])
    tr_w = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    plus_dm = np.concatenate([[np.nan], np.where((high_w[1:] - high_w[:-1]) > (low_w[:-1] - low_w[1:]), 
                                                  np.maximum(high_w[1:] - high_w[:-1], 0), 0)])
    minus_dm = np.concatenate([[np.nan], np.where((low_w[:-1] - low_w[1:]) > (high_w[1:] - high_w[:-1]), 
                                                   np.maximum(low_w[:-1] - low_w[1:], 0), 0)])
    
    # Smooth TR, +DM, -DM using Wilder's smoothing (alpha=1/14)
    def wilder_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            result[period-1] = np.nanmean(data[:period])
            for i in range(period, len(data)):
                if not np.isnan(result[i-1]) and not np.isnan(data[i]):
                    result[i] = result[i-1] * (1 - 1/period) + data[i] * (1/period)
                else:
                    result[i] = np.nan
        return result
    
    atr_w = wilder_smooth(tr_w, 14)
    plus_di_w = 100 * wilder_smooth(plus_dm, 14) / atr_w
    minus_di_w = 100 * wilder_smooth(minus_dm, 14) / atr_w
    dx_w = 100 * np.abs(plus_di_w - minus_di_w) / (plus_di_w + minus_di_w)
    adx_w = wilder_smooth(dx_w, 14)
    
    # Align weekly ADX to 12h timeframe
    adx_w_aligned = align_htf_to_ltf(prices, df_1w, adx_w)
    
    # Calculate 20-period average volume for confirmation (daily)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC
    hour_index = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or
            np.isnan(adx_w_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        hour = hour_index[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume above average
        vol_confirm = volume[i] > vol_ma_20[i]
        
        # Trend filter: weekly ADX > 25 indicates trending market
        trend_filter = adx_w_aligned[i] > 25
        
        if position == 0:
            # Long: price breaks above H3 AND volume confirmation AND trend filter
            long_breakout = close[i] > h3_aligned[i]
            if vol_confirm and trend_filter and long_breakout:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below L3 AND volume confirmation AND trend filter
            elif vol_confirm and trend_filter and close[i] < l3_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price falls below L3 OR ADX drops below 20 (trend weakening)
            exit_condition = close[i] < l3_aligned[i] or adx_w_aligned[i] < 20
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price rises above H3 OR ADX drops below 20 (trend weakening)
            exit_condition = close[i] > h3_aligned[i] or adx_w_aligned[i] < 20
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals