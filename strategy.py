#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R1/S1 breakout with 12h EMA50 trend filter and volume spike confirmation
# Long when price breaks above 1d Camarilla R1 level AND 12h close > 12h EMA50 AND volume > 2.0x 20-period average
# Short when price breaks below 1d Camarilla S1 level AND 12h close < 12h EMA50 AND volume > 2.0x 20-period average
# Exit when price crosses 12h EMA50 (trend reversal) OR price retouches the 1d Camarilla pivot point (mean reversion)
# Uses 4h primary timeframe with 1d HTF for Camarilla levels and 12h HTF for trend filter
# Camarilla levels provide mathematically derived support/resistance with high probability reversal/breakout zones
# Volume spike confirmation reduces false breakouts
# Discrete sizing (0.25) to limit fee drag and manage drawdown
# Target: 75-150 total trades over 4 years (19-37/year) for 4h timeframe

name = "4h_Camarilla_R1S1_Breakout_12hEMA50_Trend_Volume"
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
    
    # Get 1d data ONCE before loop for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Get 12h data ONCE before loop for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 1d Camarilla levels (based on previous day's OHLC)
    # Camarilla levels: 
    # R4 = close + ((high - low) * 1.1/2)
    # R3 = close + ((high - low) * 1.1/4)
    # R2 = close + ((high - low) * 1.1/6)
    # R1 = close + ((high - low) * 1.1/12)
    # PP = (high + low + close) / 3
    # S1 = close - ((high - low) * 1.1/12)
    # S2 = close - ((high - low) * 1.1/6)
    # S3 = close - ((high - low) * 1.1/4)
    # S4 = close - ((high - low) * 1.1/2)
    if len(df_1d) >= 2:
        # Use previous day's OHLC to calculate today's Camarilla levels (no look-ahead)
        prev_high = df_1d['high'].shift(1).values
        prev_low = df_1d['low'].shift(1).values
        prev_close = df_1d['close'].shift(1).values
        
        # Calculate Camarilla levels for each day based on previous day
        camarilla_pp = (prev_high + prev_low + prev_close) / 3
        camarilla_r1 = prev_close + ((prev_high - prev_low) * 1.1 / 12)
        camarilla_s1 = prev_close - ((prev_high - prev_low) * 1.1 / 12)
        
        # Align to 4h timeframe
        camarilla_pp_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pp)
        camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
        camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    else:
        camarilla_pp_aligned = np.full(n, np.nan)
        camarilla_r1_aligned = np.full(n, np.nan)
        camarilla_s1_aligned = np.full(n, np.nan)
    
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
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(camarilla_pp_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Camarilla R1 AND 12h close > 12h EMA50 AND volume spike
            if (close[i] > camarilla_r1_aligned[i] and 
                close[i] > ema_50_12h_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Camarilla S1 AND 12h close < 12h EMA50 AND volume spike
            elif (close[i] < camarilla_s1_aligned[i] and 
                  close[i] < ema_50_12h_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below 12h EMA50 (trend reversal) OR price retouches Camarilla pivot (mean reversion)
            if close[i] < ema_50_12h_aligned[i] or abs(close[i] - camarilla_pp_aligned[i]) < 0.001 * camarilla_pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above 12h EMA50 (trend reversal) OR price retouches Camarilla pivot (mean reversion)
            if close[i] > ema_50_12h_aligned[i] or abs(close[i] - camarilla_pp_aligned[i]) < 0.001 * camarilla_pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals