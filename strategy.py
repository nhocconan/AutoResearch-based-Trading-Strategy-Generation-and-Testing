#!/usr/bin/env python3
"""
Hypothesis: 4h Williams %R reversal with 12h EMA trend filter and volume spike confirmation.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 12h for EMA trend direction and volume average.
- Williams %R(14): Measures overbought/oversold levels (-20 to -80 range).
- Regime: 12h EMA50 slope > 0 = uptrend, < 0 = downtrend (avoid choppy markets).
- Entry: Long when Williams %R crosses above -80 FROM BELOW AND uptrend AND volume > 1.5 * 20-period average volume.
         Short when Williams %R crosses below -20 FROM ABOVE AND downtrend AND volume > 1.5 * 20-period average volume.
- Exit: Opposite Williams %R level (long exit when %R > -20, short exit when %R < -80).
- Signal size: 0.25 discrete to minimize fee drag.
- Works in both bull and bear markets by trading mean reversions within the trend, avoiding counter-trend whipsaws.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need sufficient data for calculations
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 12h EMA(50) for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:  # Need sufficient data for EMA50
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # EMA(50) calculation
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # EMA slope for trend: > 0 = uptrend, < 0 = downtrend
    ema_slope = np.diff(ema50_12h, prepend=ema50_12h[0])
    
    # Align EMA slope to 4h timeframe
    ema_slope_aligned = align_htf_to_ltf(prices, df_12h, ema_slope)
    
    # Calculate 12h volume average for confirmation (20-period)
    if len(df_12h) < 20:
        return np.zeros(n)
    
    vol_ma_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20_12h)
    
    # Calculate 4h Williams %R(14)
    williams_window = 14
    highest_high = pd.Series(high).rolling(window=williams_window, min_periods=williams_window).max().values
    lowest_low = pd.Series(low).rolling(window=williams_window, min_periods=williams_window).min().values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    williams_r = np.where(
        (highest_high - lowest_low) != 0,
        ((highest_high - close) / (highest_high - lowest_low)) * -100,
        -50  # neutral when range is zero
    )
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(williams_window, 50)  # Need 14 for Williams %R, 50 for EMA50
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema_slope_aligned[i]) or np.isnan(vol_ma_20_12h_aligned[i]) or
            np.isnan(williams_r[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        prev_williams_r = williams_r[i-1] if i > 0 else williams_r[i]
        
        # Trend filter: uptrend if EMA slope > 0, downtrend if EMA slope < 0
        uptrend = ema_slope_aligned[i] > 0
        downtrend = ema_slope_aligned[i] < 0
        
        # Volume confirmation: current volume > 1.5 * 20-period average volume
        volume_confirm = curr_volume > 1.5 * vol_ma_20_12h_aligned[i] if not np.isnan(vol_ma_20_12h_aligned[i]) else False
        
        # Williams %R levels
        williams_oversold = -80  # buy signal when crossing above FROM BELOW
        williams_overbought = -20  # sell signal when crossing below FROM ABOVE
        
        # Exit conditions: opposite Williams %R level
        if position != 0:
            # Exit long: Williams %R > -20 (overbought)
            if position == 1:
                if williams_r[i] > williams_overbought:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: Williams %R < -80 (oversold)
            elif position == -1:
                if williams_r[i] < williams_oversold:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Williams %R reversal with trend and volume filters
        if position == 0:
            # Long: Williams %R crosses above -80 FROM BELOW AND uptrend AND volume confirmation
            williams_cross_up = (prev_williams_r <= williams_oversold and williams_r[i] > williams_oversold)
            long_condition = williams_cross_up and uptrend and volume_confirm
            
            # Short: Williams %R crosses below -20 FROM ABOVE AND downtrend AND volume confirmation
            williams_cross_down = (prev_williams_r >= williams_overbought and williams_r[i] < williams_overbought)
            short_condition = williams_cross_down and downtrend and volume_confirm
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "4h_WilliamsR_Reversal_12hEMATrend_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0