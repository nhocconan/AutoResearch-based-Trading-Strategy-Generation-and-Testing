#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation
# Long when: price > R3 (1d Camarilla) AND 1d EMA34 rising AND volume > 1.5x 20-period MA
# Short when: price < S3 (1d Camarilla) AND 1d EMA34 falling AND volume > 1.5x 20-period MA
# Exit when: price crosses Camarilla H3/L3 levels (mean reversion) OR volume drops below average
# Uses Camarilla for intraday structure, 1d EMA for trend, volume for conviction
# Timeframe: 6h, HTF: 1d. Target: 75-150 total trades over 4 years (19-37/year) to balance opportunity and fee drag.

name = "6h_Camarilla_R3S3_Breakout_1dEMA34_VolumeConfirm"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla pivot levels on 1d
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Camarilla levels: based on previous day's OHLC
    # H4 = close + 1.5*(high-low), L4 = close - 1.5*(high-low)
    # H3 = close + 1.125*(high-low), L3 = close - 1.125*(high-low)
    # H2 = close + 0.75*(high-low), L2 = close - 0.75*(high-low)
    # H1 = close + 0.5*(high-low), L1 = close - 0.5*(high-low)
    # Pivot = (high + low + close)/3
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each 1d bar
    camarilla_h3 = np.full(len(close_1d), np.nan)
    camarilla_l3 = np.full(len(close_1d), np.nan)
    camarilla_h3_l3 = np.full(len(close_1d), np.nan)  # H3
    camarilla_l3_l3 = np.full(len(close_1d), np.nan)  # L3
    
    for i in range(1, len(close_1d)):
        # Use previous day's OHLC to calculate today's levels
        prev_high = high_1d[i-1]
        prev_low = low_1d[i-1]
        prev_close = close_1d[i-1]
        
        if not (np.isnan(prev_high) or np.isnan(prev_low) or np.isnan(prev_close)):
            range_ = prev_high - prev_low
            camarilla_h3[i] = prev_close + 1.125 * range_
            camarilla_l3[i] = prev_close - 1.125 * range_
    
    # Align Camarilla levels to 6h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Calculate EMA34 on 1d close
    if len(close_1d) >= 34:
        ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    else:
        ema_34_1d = np.full(len(close_1d), np.nan)
    
    # EMA rising/falling
    ema_rising = np.zeros(len(ema_34_1d), dtype=bool)
    ema_falling = np.zeros(len(ema_34_1d), dtype=bool)
    for i in range(1, len(ema_34_1d)):
        if not np.isnan(ema_34_1d[i]) and not np.isnan(ema_34_1d[i-1]):
            ema_rising[i] = ema_34_1d[i] > ema_34_1d[i-1]
            ema_falling[i] = ema_34_1d[i] < ema_34_1d[i-1]
    
    # Align EMA trend to 6h timeframe
    ema_rising_aligned = align_htf_to_ltf(prices, df_1d, ema_rising.astype(float))
    ema_falling_aligned = align_htf_to_ltf(prices, df_1d, ema_falling.astype(float))
    
    # Volume confirmation on 6h
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.5 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(ema_rising_aligned[i]) or np.isnan(ema_falling_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: break above Camarilla H3 + EMA rising + volume filter
            if (close[i] > camarilla_h3_aligned[i] and 
                ema_rising_aligned[i] == 1.0 and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: break below Camarilla L3 + EMA falling + volume filter
            elif (close[i] < camarilla_l3_aligned[i] and 
                  ema_falling_aligned[i] == 1.0 and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below Camarilla L3 (mean reversion) OR volume drops
            if (close[i] < camarilla_l3_aligned[i] or not volume_filter[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above Camarilla H3 (mean reversion) OR volume drops
            if (close[i] > camarilla_h3_aligned[i] or not volume_filter[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals