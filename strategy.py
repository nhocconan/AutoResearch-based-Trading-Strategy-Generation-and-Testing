#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R reversal with 1d ADX regime filter and volume confirmation.
# Williams %R identifies overbought/oversold conditions; ADX > 25 filters for trending markets to avoid false reversals in ranging markets.
# Long when Williams %R crosses above -80 (from oversold) AND 1d ADX > 25 AND volume > 1.5x 20-bar average.
# Short when Williams %R crosses below -20 (from overbought) AND 1d ADX > 25 AND volume > 1.5x 20-bar average.
# Uses discrete sizing 0.25 to manage drawdown. Target: 50-150 total trades over 4 years.
# Williams %R period = 14, ADX period = 14.

name = "12h_WilliamsR_ADX_VolumeReversal_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough for ADX calculation
        return np.zeros(n)
    
    # 1d ADX calculation (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    # Directional Movement
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Smooth TR, +DM, -DM using Wilder's smoothing (equivalent to EMA with alpha=1/14)
    def wilder_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(data[:period])
        # Subsequent values: smoothed = prev_smoothed - (prev_smoothed/period) + current
        for i in range(period, len(data)):
            if not np.isnan(result[i-1]) and not np.isnan(data[i]):
                result[i] = result[i-1] - (result[i-1]/period) + data[i]
        return result
    
    atr = wilder_smooth(tr, 14)
    plus_di = 100 * wilder_smooth(plus_dm, 14) / atr
    minus_di = 100 * wilder_smooth(minus_dm, 14) / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilder_smooth(dx, 14)
    
    # Align 1d ADX to 12h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # 1d trend: ADX > 25 indicates strong trend
    strong_trend = adx_aligned > 25
    
    # Williams %R calculation (14-period) on 12h timeframe
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # Williams %R signals: cross above -80 (long) or below -20 (short)
    williams_r_long = (williams_r[:-1] <= -80) & (williams_r[1:] > -80)
    williams_r_short = (williams_r[:-1] >= -20) & (williams_r[1:] < -20)
    # Shift to align with current bar (signal at bar i based on cross between i-1 and i)
    williams_r_long = np.concatenate([[False], williams_r_long])
    williams_r_short = np.concatenate([[False], williams_r_short])
    
    # Volume confirmation: current 12h volume > 1.5x 20-bar average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for Williams %R, ADX and volume MA
    
    for i in range(start_idx, n):
        if np.isnan(adx_aligned[i]) or np.isnan(williams_r[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        if vol_ma[i] <= 0:
            signals[i] = 0.0
            continue
            
        volume_confirm = volume[i] > (vol_ma[i] * 1.5)  # Volume threshold
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: Williams %R crosses above -80 AND 1d ADX > 25 AND volume confirmation
            if (williams_r_long[i] and 
                strong_trend[i] and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 AND 1d ADX > 25 AND volume confirmation
            elif (williams_r_short[i] and 
                  strong_trend[i] and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Williams %R crosses below -50 (momentum loss) OR ADX < 20 (trend weakening)
            if (williams_r[i] < -50 and williams_r[i-1] >= -50) or \
               adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Williams %R crosses above -50 (momentum loss) OR ADX < 20 (trend weakening)
            if (williams_r[i] > -50 and williams_r[i-1] <= -50) or \
               adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals