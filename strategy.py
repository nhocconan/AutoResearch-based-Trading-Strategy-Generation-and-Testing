#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R reversal with 1w ADX regime filter and volume confirmation.
# Uses 1w ADX > 20 to identify strong trends, reducing whipsaws in ranging markets.
# Long when Williams %R crosses above -80 (oversold reversal) AND 1w ADX > 20 AND volume > 2.0x 20-bar average.
# Short when Williams %R crosses below -20 (overbought reversal) AND 1w ADX > 20 AND volume > 2.0x 20-bar average.
# Uses discrete sizing 0.25 to manage drawdown. Target: 50-150 total trades over 4 years.
# Volume spike threshold set to 2.0x to ensure high-quality signals and minimize fee drag.
# Designed to work in both bull (trend continuation on pullbacks) and bear (counter-trend reversals) markets.

name = "12h_WilliamsR_ADX20_VolumeSpike_v1"
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
    
    # Load 1w data ONCE before loop for ADX trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:  # Need enough for ADX calculation
        return np.zeros(n)
    
    # 1w ADX calculation (14-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w[1:] - low_1w[1:]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    # Directional Movement
    up_move = high_1w[1:] - high_1w[:-1]
    down_move = low_1w[:-1] - low_1w[1:]
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
    
    # Align 1w ADX to 12h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # 1w trend: ADX > 20 indicates strong trend
    strong_trend = adx_aligned > 20
    
    # Calculate Williams %R on 1w timeframe (14-period)
    highest_high_1w = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    lowest_low_1w = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_1w - close_1w) / (highest_high_1w - lowest_low_1w)
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high_1w - lowest_low_1w) == 0, -50, williams_r)
    
    # Align Williams %R to 12h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1w, williams_r)
    
    # Williams %R signals: cross above -80 (long), cross below -20 (short)
    williams_r_long_signal = (williams_r_aligned > -80) & (np.roll(williams_r_aligned, 1) <= -80)
    williams_r_short_signal = (williams_r_aligned < -20) & (np.roll(williams_r_aligned, 1) >= -20)
    
    # Volume confirmation: current 12h volume > 2.0x 20-bar average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for ADX, Williams %R, and volume MA
    
    for i in range(start_idx, n):
        if np.isnan(williams_r_aligned[i]) or np.isnan(adx_aligned[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        curr_vol = volume[i]
        curr_vol_ma = vol_ma[i]
        
        if curr_vol_ma <= 0:
            signals[i] = 0.0
            continue
            
        volume_confirm = curr_vol > (curr_vol_ma * 2.0)  # Volume spike threshold
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: Williams %R crosses above -80 AND 1w ADX > 20 AND volume confirmation
            if (williams_r_long_signal[i] and 
                strong_trend[i] and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 AND 1w ADX > 20 AND volume confirmation
            elif (williams_r_short_signal[i] and 
                  strong_trend[i] and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Williams %R crosses below -50 (momentum loss) OR ADX < 15 (trend weakening)
            if (williams_r_aligned[i] < -50 and np.roll(williams_r_aligned, 1)[i] >= -50) or \
               adx_aligned[i] < 15:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Williams %R crosses above -50 (momentum loss) OR ADX < 15 (trend weakening)
            if (williams_r_aligned[i] > -50 and np.roll(williams_r_aligned, 1)[i] <= -50) or \
               adx_aligned[i] < 15:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals