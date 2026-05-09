#!/usr/bin/env python3
# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike
# Long when: close > Camarilla R3, 1d EMA34 rising, volume spike (>2x 20-period average)
# Short when: close < Camarilla S3, 1d EMA34 falling, volume spike
# Exit when: price crosses Camarilla H/L levels OR trend reverses
# Position size: 0.25 to limit drawdown. Target: 20-40 trades/year.
# Designed to work in both bull (breakouts) and bear (mean-reversion at extremes) markets.
# Uses proven Camarilla pivot structure with volume confirmation and trend filter.

name = "4h_Camarilla_R3S3_1dEMA34_VolumeSpike"
timeframe = "4h"
leverage = 1.0

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
    
    # Calculate Camarilla levels (based on previous day's range)
    # Camarilla: H4 = C + 1.1*(H-L)/2, L4 = C - 1.1*(H-L)/2
    # R3 = C + 1.1*(H-L)/2, S3 = C - 1.1*(H-L)/2
    # We'll use daily high/low/close to calculate intraday levels
    # Since we're on 4h timeframe, we need to get daily OHLC
    
    # Get daily data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's data
    # We need to shift by 1 to avoid look-ahead (use previous day's data)
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Calculate Camarilla R3 and S3 levels
    camarilla_r3 = prev_close + 1.1 * (prev_high - prev_low) / 2
    camarilla_s3 = prev_close - 1.1 * (prev_high - prev_low) / 2
    
    # Align Camarilla levels to 4h timeframe (they change only at daily boundaries)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Also calculate H3 and L3 for exit levels (optional)
    camarilla_h3 = prev_close + 1.1 * (prev_high - prev_low) / 4
    camarilla_l3 = prev_close - 1.1 * (prev_high - prev_low) / 4
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Get 1d EMA34 for trend filter
    close_1d = df_1d['close']
    ema_34_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_prev = np.roll(ema_34_1d, 1)
    ema_34_1d_prev[0] = ema_34_1d[0]
    ema_rising = ema_34_1d > ema_34_1d_prev
    ema_falling = ema_34_1d < ema_34_1d_prev
    ema_rising_aligned = align_htf_to_ltf(prices, df_1d, ema_rising)
    ema_falling_aligned = align_htf_to_ltf(prices, df_1d, ema_falling)
    
    # Volume spike: current volume > 2x 20-period average volume (more selective)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_spike = volume > (2.0 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(ema_rising_aligned[i]) or np.isnan(ema_falling_aligned[i]) or
            np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price > Camarilla R3 + 1d EMA34 rising + volume spike
            if (close[i] > camarilla_r3_aligned[i] and 
                ema_rising_aligned[i] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price < Camarilla S3 + 1d EMA34 falling + volume spike
            elif (close[i] < camarilla_s3_aligned[i] and 
                  ema_falling_aligned[i] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below Camarilla L3 OR trend turns down
            if (close[i] < camarilla_l3_aligned[i]) or (not ema_rising_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above Camarilla H3 OR trend turns up
            if (close[i] > camarilla_h3_aligned[i]) or (not ema_falling_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals