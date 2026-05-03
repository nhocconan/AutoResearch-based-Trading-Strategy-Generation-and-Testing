#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot R3/S3 breakout with 1d volume spike and 1w EMA34 trend filter.
# Long when price breaks above Camarilla R3 AND 1d volume > 2.0x 20-period volume MA AND 1w close > 1w EMA34 (uptrend).
# Short when price breaks below Camarilla S3 AND 1d volume > 2.0x 20-period volume MA AND 1w close < 1w EMA34 (downtrend).
# Uses session filter (08-20 UTC) to avoid low-liquidity periods. Position size fixed at 0.25.
# Designed for 4h timeframe to achieve 75-200 total trades over 4 years (19-50/year) with strict entry conditions.
# Camarilla levels provide objective intraday support/resistance, 1d volume confirms institutional participation, 1w EMA34 filters for higher timeframe trend alignment.
# Works in both bull and bear markets by only trading breakouts in the direction of the 1w trend when volume confirms.

name = "4h_Camarilla_R3S3_Breakout_1dVolumeSpike_1wEMA34_Trend_Session"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for volume confirmation and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d volume 20-period MA for spike detection
    volume_ma_1d = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    volume_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_1d)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 1w EMA34 for trend direction
    ema_34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate 1d OHLC for Camarilla levels (using previous day's data)
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Camarilla levels: R3 = C + (H-L)*1.1/4, S3 = C - (H-L)*1.1/4
    camarilla_r3 = prev_close + (prev_high - prev_low) * 1.1 / 4
    camarilla_s3 = prev_close - (prev_high - prev_low) * 1.1 / 4
    
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(volume_ma_1d_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        
        # Volume spike condition: 1d volume > 2.0x 20-period MA
        # Use current 1d volume (approximated as last value) vs aligned MA
        vol_1d_current = df_1d['volume'].iloc[-1] if len(df_1d) > 0 else 0
        volume_spike = vol_1d_current > (volume_ma_1d_aligned[i] * 2.0)
        
        # Trend conditions
        trend_up = close_val > ema_34_1w_aligned[i]   # 1w uptrend
        trend_down = close_val < ema_34_1w_aligned[i]  # 1w downtrend
        
        if position == 0:
            # Long: Camarilla R3 breakout AND 1w uptrend AND volume spike AND session
            if close_val > camarilla_r3_aligned[i] and trend_up and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: Camarilla S3 breakdown AND 1w downtrend AND volume spike AND session
            elif close_val < camarilla_s3_aligned[i] and trend_down and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price retouches Camarilla H3/L3 level OR trend changes
            camarilla_h3 = camarilla_r3_aligned[i] - (camarilla_r3_aligned[i] - camarilla_s3_aligned[i]) * 0.125
            camarilla_l3 = camarilla_s3_aligned[i] + (camarilla_r3_aligned[i] - camarilla_s3_aligned[i]) * 0.125
            if close_val < camarilla_h3 or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price retouches Camarilla H3/L3 level OR trend changes
            camarilla_h3 = camarilla_r3_aligned[i] - (camarilla_r3_aligned[i] - camarilla_s3_aligned[i]) * 0.125
            camarilla_l3 = camarilla_s3_aligned[i] + (camarilla_r3_aligned[i] - camarilla_s3_aligned[i]) * 0.125
            if close_val > camarilla_l3 or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals