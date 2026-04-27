#!/usr/bin/env python3
"""
1d_Camarilla_R1_S1_Breakout_1wTrend_VolumeSpike_v1
Hypothesis: Daily Camarilla R1/S1 breakout with 1-week EMA34 trend filter and volume confirmation.
Breakouts above/below Camarilla R1/S1 levels capture strong intraday momentum moves.
Trend filter ensures we only trade in direction of weekly trend to avoid counter-trend whipsaws.
Volume spike confirms breakout authenticity. Designed for 1d timeframe with target 30-100 trades over 4 years.
Uses discrete position sizing (0.25) to balance return and drawdown. Works in both bull and bear markets
by aligning with longer-term weekly trend, reducing whipsaws during sideways periods.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla levels for daily timeframe using previous day's OHLC
    # Camarilla: R1 = close + (high - low) * 1.1/12, S1 = close - (high - low) * 1.1/12
    # We need to shift by 1 to avoid look-ahead (use previous day's range)
    prev_high = np.concatenate([[high[0]], high[:-1]])
    prev_low = np.concatenate([[low[0]], low[:-1]])
    prev_close = np.concatenate([[close[0]], close[:-1]])
    
    rang = prev_high - prev_low
    camarilla_r1 = prev_close + rang * 1.1 / 12
    camarilla_s1 = prev_close - rang * 1.1 / 12
    
    # Calculate 1w EMA34 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: need enough for Camarilla (shifted), EMA34 and volume average
    start_idx = max(100, 34, 20, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_r1[i]) or np.isnan(camarilla_s1[i]) or
            np.isnan(ema_34_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema_trend = ema_34_aligned[i]
        vol_spike = volume_spike[i]
        size = 0.25  # 25% position size
        
        if position == 0:
            # Flat - look for entry: breakout in direction of 1w trend with volume spike
            # Long: price breaks above Camarilla R1 AND 1w trend is up (close > EMA34) AND volume spike
            # Short: price breaks below Camarilla S1 AND 1w trend is down (close < EMA34) AND volume spike
            long_breakout = close_val > camarilla_r1[i]
            short_breakout = close_val < camarilla_s1[i]
            trend_up = close_val > ema_trend
            trend_down = close_val < ema_trend
            
            if long_breakout and trend_up and vol_spike:
                signals[i] = size
                position = 1
                entry_price = close_val
            elif short_breakout and trend_down and vol_spike:
                signals[i] = -size
                position = -1
                entry_price = close_val
        elif position == 1:
            # Long - exit when price breaks below Camarilla S1 (failed breakout) or reverse below EMA34
            if close_val < camarilla_s1[i] or close_val < ema_trend:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit when price breaks above Camarilla R1 (failed breakout) or reverse above EMA34
            if close_val > camarilla_r1[i] or close_val > ema_trend:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "1d_Camarilla_R1_S1_Breakout_1wTrend_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0