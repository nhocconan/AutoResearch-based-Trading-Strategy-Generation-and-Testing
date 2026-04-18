#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Daily Range Breakout with Weekly ADX Trend Filter and Volume Confirmation.
# Uses the previous day's high/low as breakout levels, traded only when weekly ADX > 25 (trending market).
# Volume confirmation ensures conviction. Designed for low frequency (10-30 trades/year) on 1d timeframe.
# Works in bull markets (breakouts above prior day high) and bear markets (breakouts below prior day low).
# Weekly ADX filter avoids choppy markets where breakouts fail.
name = "1d_DailyRangeBreakout_WeeklyADX_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 10:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for ADX filter (ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate prior day's high and low for breakout levels (shifted by 1 to avoid look-ahead)
    # Using 1-day period since we're on 1d timeframe
    prior_high = np.roll(high, 1)  # previous day's high
    prior_low = np.roll(low, 1)    # previous day's low
    prior_high[0] = np.nan         # first day has no prior
    prior_low[0] = np.nan
    
    # Calculate weekly ADX (14-period)
    high_w = df_1w['high'].values
    low_w = df_1w['low'].values
    close_w = df_1w['close'].values
    
    # True Range calculation for weekly
    tr1 = high_w[1:] - low_w[1:]
    tr2 = np.abs(high_w[1:] - close_w[:-1])
    tr3 = np.abs(low_w[1:] - close_w[:-1])
    tr_w = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    plus_dm = np.concatenate([[np.nan], np.maximum(high_w[1:] - high_w[:-1], 0)])
    minus_dm = np.concatenate([[np.nan], np.maximum(low_w[:-1] - low_w[1:], 0)])
    
    # Wilder's smoothing for TR, +DM, -DM
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            result[period-1] = np.nanmean(data[:period])
            for i in range(period, len(data)):
                if not np.isnan(result[i-1]) and not np.isnan(data[i]):
                    result[i] = result[i-1] * (1 - 1/period) + data[i] * (1/period)
                else:
                    result[i] = np.nan
        return result
    
    atr_w = wilders_smoothing(tr_w, 14)
    plus_dm_w = wilders_smoothing(plus_dm, 14)
    minus_dm_w = wilders_smoothing(minus_dm, 14)
    
    # Directional Indicators
    plus_di = 100 * plus_dm_w / atr_w
    minus_di = 100 * minus_dm_w / atr_w
    
    # DX and ADX
    dx = np.full_like(atr_w, np.nan)
    mask = (plus_di + minus_di) != 0
    dx[mask] = 100 * np.abs(plus_di[mask] - minus_di[mask]) / (plus_di[mask] + minus_di[mask])
    
    adx = wilders_smoothing(dx, 14)
    
    # Align weekly ADX to daily timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # Calculate 20-day average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Wait for volume MA calculation
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(prior_high[i]) or np.isnan(prior_low[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: weekly ADX > 25 indicates trending market
        trend_filter = adx_aligned[i] > 25
        
        # Volume confirmation: current volume above average
        vol_confirm = volume[i] > vol_ma_20[i]
        
        if position == 0:
            # Long: price breaks above prior day's high AND trend filter AND volume confirmation
            long_breakout = close[i] > prior_high[i]
            if trend_filter and vol_confirm and long_breakout:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below prior day's low AND trend filter AND volume confirmation
            elif trend_filter and vol_confirm and close[i] < prior_low[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price falls below prior day's low
            if close[i] < prior_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price rises above prior day's high
            if close[i] > prior_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals