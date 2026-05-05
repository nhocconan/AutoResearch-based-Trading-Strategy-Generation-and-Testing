#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 12h volume confirmation and 12h EMA34 trend filter
# Long when price breaks above 4h Camarilla R3 level AND 12h volume > 2.0x 20-period average AND close > 12h EMA34
# Short when price breaks below 4h Camarilla S3 level AND 12h volume > 2.0x 20-period average AND close < 12h EMA34
# Exit when price crosses 4h Camarilla midpoint (mean reversion)
# Uses 4h primary timeframe with 12h HTF for volume and trend confirmation
# Camarilla levels provide institutional support/resistance; volume confirms breakout validity; EMA filter ensures trend alignment
# Discrete sizing (0.25) to limit fee drag and manage drawdown
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe

name = "4h_Camarilla_R3S3_Breakout_12hVolume_12hEMA34"
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
    
    # Get 12h data ONCE before loop for volume and trend confirmation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 40:
        return np.zeros(n)
    
    # Calculate 12h volume spike filter
    vol_12h = df_12h['volume'].values
    if len(vol_12h) >= 20:
        vol_ma_20 = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
        volume_filter_12h = vol_12h > (2.0 * vol_ma_20)
    else:
        volume_filter_12h = np.zeros(len(df_12h), dtype=bool)
    
    # Calculate 12h EMA34 trend filter
    close_12h = df_12h['close'].values
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Get 4h data ONCE before loop for Camarilla levels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Calculate 4h Camarilla levels (based on previous day's OHLC)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    open_4h = df_4h['open'].values
    
    # Camarilla levels use previous day's range
    # For intraday, we use the previous 4h bar's OHLC to simulate daily calculation
    # Camarilla formulas: 
    # R4 = close + (high - low) * 1.1/2
    # R3 = close + (high - low) * 1.1/4
    # R2 = close + (high - low) * 1.1/6
    # R1 = close + (high - low) * 1.1/12
    # PP = (high + low + close) / 3
    # S1 = close - (high - low) * 1.1/12
    # S2 = close - (high - low) * 1.1/6
    # S3 = close - (high - low) * 1.1/4
    # S4 = close - (high - low) * 1.1/2
    
    # Calculate using previous bar's OHLC (shifted by 1 to avoid look-ahead)
    prev_high = np.roll(high_4h, 1)
    prev_low = np.roll(low_4h, 1)
    prev_close = np.roll(close_4h, 1)
    prev_high[0] = prev_low[0] = prev_close[0] = np.nan  # First bar has no previous
    
    camarilla_pp = (prev_high + prev_low + prev_close) / 3
    camarilla_range = prev_high - prev_low
    camarilla_r3 = camarilla_pp + camarilla_range * 1.1 / 4
    camarilla_s3 = camarilla_pp - camarilla_range * 1.1 / 4
    camarilla_mid = camarilla_pp  # Using pivot point as midpoint for exit
    
    # Align 12h indicators to 4h timeframe
    volume_filter_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_filter_12h)
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Align 4h Camarilla levels to 4h timeframe (same df_4h)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s3)
    camarilla_mid_aligned = align_htf_to_ltf(prices, df_4h, camarilla_mid)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(camarilla_mid_aligned[i]) or 
            np.isnan(ema_34_12h_aligned[i]) or 
            np.isnan(volume_filter_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Camarilla R3 AND volume spike AND above 12h EMA34
            if (close[i] > camarilla_r3_aligned[i] and 
                volume_filter_12h_aligned[i] and 
                close[i] > ema_34_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Camarilla S3 AND volume spike AND below 12h EMA34
            elif (close[i] < camarilla_s3_aligned[i] and 
                  volume_filter_12h_aligned[i] and 
                  close[i] < ema_34_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below Camarilla midpoint (mean reversion)
            if close[i] < camarilla_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above Camarilla midpoint (mean reversion)
            if close[i] > camarilla_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals