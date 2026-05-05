#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation
# Long when price breaks above Camarilla R3 AND price > EMA34(1d) AND volume > 2.0x 20-period average
# Short when price breaks below Camarilla S3 AND price < EMA34(1d) AND volume > 2.0x 20-period average
# Exit when price crosses back below/above Camarilla pivot point (mean reversion) OR trend flips (price crosses EMA34(1d))
# Camarilla levels provide institutional support/resistance from prior day's range
# 1d EMA34 provides higher timeframe trend filter to avoid counter-trend whipsaws
# Volume spike confirms institutional participation
# Target: 12-37 trades/year per symbol (50-150 total over 4 years) for 12h timeframe
# Discrete sizing (0.25) to limit fee drag

name = "12h_Camarilla_R3_S3_Breakout_1dEMA34_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for HTF indicators
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate EMA34 on 1d close for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA34 to 12h timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from 1d OHLC
    # Camarilla: based on previous day's range
    # R4 = close + ((high-low) * 1.1/2)
    # R3 = close + ((high-low) * 1.1/4)
    # R2 = close + ((high-low) * 1.1/6)
    # R1 = close + ((high-low) * 1.1/12)
    # PP = (high + low + close) / 3
    # S1 = close - ((high-low) * 1.1/12)
    # S2 = close - ((high-low) * 1.1/6)
    # S3 = close - ((high-low) * 1.1/4)
    # S4 = close - ((high-low) * 1.1/2)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_vals = df_1d['close'].values
    
    # Calculate Camarilla levels for each 1d bar
    camarilla_R3 = np.full_like(close_1d_vals, np.nan)
    camarilla_S3 = np.full_like(close_1d_vals, np.nan)
    camarilla_PP = np.full_like(close_1d_vals, np.nan)
    
    for i in range(1, len(close_1d_vals)):  # Start from 1 to use previous day's data
        prev_high = high_1d[i-1]
        prev_low = low_1d[i-1]
        prev_close = close_1d_vals[i-1]
        
        rang = prev_high - prev_low
        if rang <= 0:
            camarilla_R3[i] = prev_close
            camarilla_S3[i] = prev_close
            camarilla_PP[i] = prev_close
        else:
            camarilla_R3[i] = prev_close + (rang * 1.1 / 4)
            camarilla_S3[i] = prev_close - (rang * 1.1 / 4)
            camarilla_PP[i] = (prev_high + prev_low + prev_close) / 3
    
    # Align Camarilla levels to 12h timeframe (use previous day's levels for current day)
    camarilla_R3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R3)
    camarilla_S3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S3)
    camarilla_PP_aligned = align_htf_to_ltf(prices, df_1d, camarilla_PP)
    
    # Volume confirmation: volume > 2.0x 20-period average (spike filter)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (2.0 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(camarilla_R3_aligned[i]) or 
            np.isnan(camarilla_S3_aligned[i]) or 
            np.isnan(camarilla_PP_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Camarilla R3 AND price > EMA34(1d) AND volume spike
            if (close[i] > camarilla_R3_aligned[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Camarilla S3 AND price < EMA34(1d) AND volume spike
            elif (close[i] < camarilla_S3_aligned[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses back below Camarilla PP (mean reversion) OR price < EMA34(1d) (trend flip)
            if (close[i] < camarilla_PP_aligned[i] or 
                close[i] < ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses back above Camarilla PP (mean reversion) OR price > EMA34(1d) (trend flip)
            if (close[i] > camarilla_PP_aligned[i] or 
                close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals