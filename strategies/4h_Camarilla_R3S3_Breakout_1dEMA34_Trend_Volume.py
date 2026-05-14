#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation
# Long when price breaks above 4h Camarilla R3 level AND 1d close > 1d EMA34 AND volume > 2.0x 20-period average
# Short when price breaks below 4h Camarilla S3 level AND 1d close < 1d EMA34 AND volume > 2.0x 20-period average
# Exit when price crosses 1d EMA34 (trend reversal) OR price retouches the 4h Camarilla pivot point (mean reversion)
# Uses 4h primary timeframe with 1d HTF for all indicators (Camarilla levels, EMA34)
# Camarilla R3/S3 levels provide stronger breakout confirmation than R1/S1, reducing false signals
# Volume confirmation filters low-momentum breakouts
# Discrete sizing (0.25) to limit fee drag and manage drawdown
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_Trend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for all indicators
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Get 4h data ONCE before loop for Camarilla levels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) >= 2:
        # Use previous bar's OHLC to calculate current bar's Camarilla levels (no look-ahead)
        prev_high = df_4h['high'].shift(1).values
        prev_low = df_4h['low'].shift(1).values
        prev_close = df_4h['close'].shift(1).values
        
        # Calculate Camarilla levels for each bar based on previous bar
        camarilla_pp = (prev_high + prev_low + prev_close) / 3
        camarilla_r3 = prev_close + ((prev_high - prev_low) * 1.1 / 4)
        camarilla_s3 = prev_close - ((prev_high - prev_low) * 1.1 / 4)
        
        # Align to 4h timeframe
        camarilla_pp_aligned = align_htf_to_ltf(prices, df_4h, camarilla_pp)
        camarilla_r3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r3)
        camarilla_s3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s3)
    else:
        camarilla_pp_aligned = np.full(n, np.nan)
        camarilla_r3_aligned = np.full(n, np.nan)
        camarilla_s3_aligned = np.full(n, np.nan)
    
    # Volume confirmation: volume > 2.0x 20-period average
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
            np.isnan(camarilla_pp_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Camarilla R3 AND 1d close > 1d EMA34 AND volume spike
            if (close[i] > camarilla_r3_aligned[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Camarilla S3 AND 1d close < 1d EMA34 AND volume spike
            elif (close[i] < camarilla_s3_aligned[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below 1d EMA34 (trend reversal) OR price retouches Camarilla pivot (mean reversion)
            if close[i] < ema_34_1d_aligned[i] or abs(close[i] - camarilla_pp_aligned[i]) < 0.001 * camarilla_pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above 1d EMA34 (trend reversal) OR price retouches Camarilla pivot (mean reversion)
            if close[i] > ema_34_1d_aligned[i] or abs(close[i] - camarilla_pp_aligned[i]) < 0.001 * camarilla_pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals