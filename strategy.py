#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator + 1d EMA34 Trend Filter + Volume Spike
# Uses Williams Alligator (3 SMAs: Jaw, Teeth, Lips) for trend direction on 4h,
# daily EMA34 for long-term trend bias, and volume spike (>1.5x average) for entry.
# Designed to work in both bull and bear markets by aligning with the daily trend
# and using Alligator to avoid whipsaws. Target: 20-40 trades/year.

name = "4h_WilliamsAlligator_1dEMA34_VolumeSpike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for EMA trend filter
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 34:
        return np.zeros(n)
    
    # Calculate daily EMA34 for trend filter
    close_daily = df_daily['close'].values
    ema34_daily = np.full(len(close_daily), np.nan)
    if len(close_daily) >= 34:
        ema34_daily[33] = np.mean(close_daily[:34])
        for i in range(34, len(close_daily)):
            ema34_daily[i] = (close_daily[i] * 2 + ema34_daily[i-1] * 32) / 34
    
    # Calculate Williams Alligator on 4h: SMA(13,8), SMA(8,5), SMA(5,3) shifted
    # Jaw: SMA(13,8), Teeth: SMA(8,5), Lips: SMA(5,3)
    def sma(arr, window):
        res = np.full_like(arr, np.nan, dtype=np.float64)
        if len(arr) >= window:
            for i in range(window-1, len(arr)):
                res[i] = np.mean(arr[i-window+1:i+1])
        return res
    
    jaw = sma(close, 13)
    teeth = sma(close, 8)
    lips = sma(close, 5)
    
    # Shift as per Williams Alligator: Jaw by 8, Teeth by 5, Lips by 3
    jaw_shifted = np.full_like(jaw, np.nan)
    teeth_shifted = np.full_like(teeth, np.nan)
    lips_shifted = np.full_like(lips, np.nan)
    if len(jaw) > 8:
        jaw_shifted[8:] = jaw[:-8]
    if len(teeth) > 5:
        teeth_shifted[5:] = teeth[:-5]
    if len(lips) > 3:
        lips_shifted[3:] = lips[:-3]
    
    # Calculate daily volume average for volume spike
    vol_daily = df_daily['volume'].values
    vol_avg_20_daily = np.full(len(vol_daily), np.nan)
    if len(vol_daily) >= 20:
        for i in range(20, len(vol_daily)):
            vol_avg_20_daily[i] = np.mean(vol_daily[i-20:i])
    
    # Align daily indicators to 4h timeframe
    ema34_daily_aligned = align_htf_to_ltf(prices, df_daily, ema34_daily)
    vol_avg_20_daily_aligned = align_htf_to_ltf(prices, df_daily, vol_avg_20_daily)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20, 13)  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(ema34_daily_aligned[i]) or np.isnan(vol_avg_20_daily_aligned[i]) or
            np.isnan(jaw_shifted[i]) or np.isnan(teeth_shifted[i]) or np.isnan(lips_shifted[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current 4h volume > 1.5x 20-period average of daily volume
        vol_spike = False
        if not np.isnan(vol_avg_20_daily_aligned[i]):
            vol_spike = volume[i] > 1.5 * vol_avg_20_daily_aligned[i]
        
        if position == 0:
            # Look for entry: Alligator aligned + daily trend + volume spike
            # Alligator long: Lips > Teeth > Jaw (bullish alignment)
            # Alligator short: Lips < Teeth < Jaw (bearish alignment)
            alligator_long = (lips_shifted[i] > teeth_shifted[i]) and (teeth_shifted[i] > jaw_shifted[i])
            alligator_short = (lips_shifted[i] < teeth_shifted[i]) and (teeth_shifted[i] < jaw_shifted[i])
            
            # Long when Alligator bullish AND price above daily EMA34 AND volume spike
            long_condition = (
                alligator_long and
                close[i] > ema34_daily_aligned[i] and
                vol_spike
            )
            
            # Short when Alligator bearish AND price below daily EMA34 AND volume spike
            short_condition = (
                alligator_short and
                close[i] < ema34_daily_aligned[i] and
                vol_spike
            )
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator turns bearish or price crosses below daily EMA34
            if not alligator_long or close[i] < ema34_daily_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator turns bullish or price crosses above daily EMA34
            if not alligator_short or close[i] > ema34_daily_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals