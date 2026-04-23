#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R3/S3 breakout with 12h trend filter and volume confirmation.
Long when price breaks above 4h Camarilla R3 level AND 12h close > 12h EMA34 (uptrend) AND volume > 1.5x 20-period MA.
Short when price breaks below 4h Camarilla S3 level AND 12h close < 12h EMA34 (downtrend) AND volume > 1.5x 20-period MA.
Exit when price retouches 4h Camarilla pivot point or 12h trend reverses.
Camarilla levels provide precise intraday support/resistance; 12h EMA34 filters counter-trend trades; volume confirmation reduces false breakouts.
Designed for low trade frequency (target: 20-50/year) to minimize fee drag and work in both bull and bear markets via trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 4h Camarilla levels (based on previous bar's OHLC)
    # R3 = C + (H-L)*1.1/2, S3 = C - (H-L)*1.1/2, P = (H+L+C)/3
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    camarilla_pivot = (prev_high + prev_low + prev_close) / 3
    camarilla_r3 = prev_close + (prev_high - prev_low) * 1.1 / 2
    camarilla_s3 = prev_close - (prev_high - prev_low) * 1.1 / 2
    
    # Calculate 12h EMA34 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Calculate 4h volume MA (20-period) for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)  # EMA34 needs 34, volume MA needs 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or np.isnan(camarilla_pivot[i]) or 
            np.isnan(ema_34_12h_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: 12h close > EMA34 = uptrend, close < EMA34 = downtrend
        close_12h_aligned = align_htf_to_ltf(prices, df_12h, close_12h)
        trend_up = close_12h_aligned[i] > ema_34_12h_aligned[i]
        trend_down = close_12h_aligned[i] < ema_34_12h_aligned[i]
        
        # Volume filter: 4h volume > 1.5x 20-period MA
        vol_filter = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Long: price breaks above R3 AND uptrend AND volume filter
            if close[i] > camarilla_r3[i] and trend_up and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 AND downtrend AND volume filter
            elif close[i] < camarilla_s3[i] and trend_down and vol_filter:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: price retouches pivot point (close crosses below pivot) OR 12h trend turns down
                if close[i] < camarilla_pivot[i] or not trend_up:
                    exit_signal = True
            elif position == -1:
                # Short exit: price retouches pivot point (close crosses above pivot) OR 12h trend turns up
                if close[i] > camarilla_pivot[i] or not trend_down:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Camarilla_R3S3_Breakout_12hEMA34_Trend_VolumeFilter"
timeframe = "4h"
leverage = 1.0