#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R1/S1 breakout with 12h EMA34 trend and volume spike.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 12h for EMA34 trend filter, 1d for Camarilla pivot levels and volume average.
- Entry: Long when price breaks above Camarilla R1 AND price > 12h EMA34 AND volume > 2.0 * 20-period average volume.
         Short when price breaks below Camarilla S1 AND price < 12h EMA34 AND volume > 2.0 * 20-period average volume.
- Exit: Opposite Camarilla breakout signal.
- Signal size: 0.25 discrete to minimize fee drag.
- Camarilla levels provide intraday support/resistance; EMA34 filters trend direction; volume spike confirms legitimacy.
- Works in bull markets (breakouts with trend) and bear markets (mean reversion at extremes with volume).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:  # Need sufficient data for calculations
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d Camarilla pivot levels (R1, S1)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:  # Need at least 2 days for pivot calculation
        return np.zeros(n)
    
    # Camarilla levels: based on previous day's high, low, close
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Calculate Camarilla R1 and S1
    camarilla_r1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    camarilla_s1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Calculate 1d volume average for confirmation (20-period)
    if len(df_1d) < 20:
        return np.zeros(n)
    vol_ma_20 = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    # Calculate 12h EMA34 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    ema_34 = pd.Series(df_12h['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_12h, ema_34)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)  # Need 34 for EMA, 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(vol_ma_20_aligned[i]) or np.isnan(ema_34_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        prev_close = close[i-1]
        
        # Exit conditions: opposite Camarilla breakout
        if position != 0:
            # Exit long: price breaks below Camarilla S1
            if position == 1:
                if curr_low <= camarilla_s1_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price breaks above Camarilla R1
            elif position == -1:
                if curr_high >= camarilla_r1_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Camarilla breakout with volume confirmation and EMA trend filter
        if position == 0:
            # Camarilla breakout signals (using current bar's high/low for breakout)
            breakout_up = curr_high >= camarilla_r1_aligned[i] and prev_close < camarilla_r1_aligned[i]
            breakout_down = curr_low <= camarilla_s1_aligned[i] and prev_close > camarilla_s1_aligned[i]
            
            # Volume confirmation: current volume > 2.0 * 20-period average volume (aligned)
            volume_confirm = curr_volume > 2.0 * vol_ma_20_aligned[i] if not np.isnan(vol_ma_20_aligned[i]) else False
            
            # EMA trend filter: price above/below EMA34 for long/short
            ema_filter_long = curr_close > ema_34_aligned[i]
            ema_filter_short = curr_close < ema_34_aligned[i]
            
            if breakout_up and volume_confirm and ema_filter_long:
                signals[i] = 0.25
                position = 1
            elif breakout_down and volume_confirm and ema_filter_short:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_12hEMA34_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0