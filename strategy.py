#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using 1w Camarilla pivot levels (R3/S3) breakout with volume confirmation and ADX trend filter.
# Enter long when price breaks above R3 with volume > 2.0x average and ADX > 25 (strong trend).
# Enter short when price breaks below S3 with volume > 2.0x average and ADX > 25.
# Exit when price returns to the 1w pivot level (PP) or opposite Camarilla level is touched.
# Camarilla levels provide institutional support/resistance; breakouts with volume confirm institutional participation.
# ADX filter ensures we only trade in trending markets, avoiding whipsaws in ranging conditions.
# Works in bull markets (breakouts continue up) and bear markets (breakdowns continue down).
# Uses discrete position sizing (0.25) to control risk. Target: 30-100 total trades over 4 years.

name = "1d_Camarilla_R3S3_Breakout_1wADX25_Volume2x_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for Camarilla pivot calculation (HTF)
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate 1w Camarilla pivot levels
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Pivot Point (PP)
    PP = (high_1w + low_1w + close_1w) / 3.0
    # Range
    range_1w = high_1w - low_1w
    
    # Camarilla levels
    R3 = PP + range_1w * 1.1 / 4.0
    S3 = PP - range_1w * 1.1 / 4.0
    R4 = PP + range_1w * 1.1 / 2.0
    S4 = PP - range_1w * 1.1 / 2.0
    
    # Align Camarilla levels to 1d timeframe
    PP_aligned = align_htf_to_ltf(prices, df_1w, PP)
    R3_aligned = align_htf_to_ltf(prices, df_1w, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1w, S3)
    R4_aligned = align_htf_to_ltf(prices, df_1w, R4)
    S4_aligned = align_htf_to_ltf(prices, df_1w, S4)
    
    # Calculate 1d ADX (14-period)
    plus_dm = np.zeros_like(high)
    minus_dm = np.zeros_like(high)
    for i in range(1, n):
        plus_dm[i] = max(0, high[i] - high[i-1])
        minus_dm[i] = max(0, low[i-1] - low[i])
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Smoothed values (Wilder's smoothing)
    def wilders_smoothing(data, period):
        result = np.zeros_like(data)
        if len(data) < period:
            return result
        result[period-1] = np.nansum(data[:period])
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    period_adx = 14
    if n >= period_adx:
        tr_smoothed = wilders_smoothing(tr, period_adx)
        plus_dm_smoothed = wilders_smoothing(plus_dm, period_adx)
        minus_dm_smoothed = wilders_smoothing(minus_dm, period_adx)
        
        plus_di = 100 * plus_dm_smoothed / tr_smoothed
        minus_di = 100 * minus_dm_smoothed / tr_smoothed
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = wilders_smoothing(dx, period_adx)
    else:
        adx = np.full(n, np.nan)
    
    # Calculate 1d volume confirmation: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(PP_aligned[i]) or np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or
            np.isnan(adx[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # ADX trend filter: only trade when ADX > 25 (strong trend)
        strong_trend = adx[i] > 25
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        # Camarilla breakout conditions
        long_breakout = close[i] > R3_aligned[i]
        short_breakout = close[i] < S3_aligned[i]
        
        # Exit conditions: return to pivot level (PP)
        long_exit = close[i] < PP_aligned[i]
        short_exit = close[i] > PP_aligned[i]
        
        # Entry conditions
        long_entry = long_breakout and strong_trend and vol_confirm
        short_entry = short_breakout and strong_trend and vol_confirm
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif (position == 1 and long_exit) or (position == -1 and short_exit):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals