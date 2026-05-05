#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA50 trend filter and volume spike confirmation
# Long when price breaks above 4h Camarilla R1 level AND 4h close > 4h EMA50 AND volume > 2.0x 20-period average
# Short when price breaks below 4h Camarilla S1 level AND 4h close < 4h EMA50 AND volume > 2.0x 20-period average
# Exit when price crosses 4h EMA50 (trend reversal) OR price retouches 4h Camarilla pivot point (mean reversion)
# Uses 1h primary timeframe with 4h HTF for Camarilla levels, trend, and volume confirmation
# Session filter: 08-20 UTC to avoid low-liquidity periods
# Discrete sizing (0.20) to limit fee drag and manage drawdown
# Target: 60-150 total trades over 4 years = 15-37/year for 1h timeframe

name = "1h_Camarilla_R1S1_Breakout_4hEMA50_Trend_Volume"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Precompute session hours (08-20 UTC) once before loop
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data ONCE before loop for Camarilla levels, trend filter, and volume confirmation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 4h Camarilla levels (based on previous 4h bar's OHLC)
    if len(df_4h) >= 2:
        # Use previous 4h bar's OHLC to calculate current 4h bar's Camarilla levels (no look-ahead)
        prev_high = df_4h['high'].shift(1).values
        prev_low = df_4h['low'].shift(1).values
        prev_close = df_4h['close'].shift(1).values
        
        # Calculate Camarilla levels for each 4h bar based on previous 4h bar
        camarilla_pp = (prev_high + prev_low + prev_close) / 3
        camarilla_r1 = prev_close + ((prev_high - prev_low) * 1.1 / 12)
        camarilla_s1 = prev_close - ((prev_high - prev_low) * 1.1 / 12)
        
        # Align to 1h timeframe
        camarilla_pp_aligned = align_htf_to_ltf(prices, df_4h, camarilla_pp)
        camarilla_r1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r1)
        camarilla_s1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s1)
    else:
        camarilla_pp_aligned = np.full(n, np.nan)
        camarilla_r1_aligned = np.full(n, np.nan)
        camarilla_s1_aligned = np.full(n, np.nan)
    
    # Volume confirmation: volume > 2.0x 20-period average (using 1h volume)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (2.0 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(camarilla_pp_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(volume_filter[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Camarilla R1 AND 4h close > 4h EMA50 AND volume spike
            if (close[i] > camarilla_r1_aligned[i] and 
                close[i] > ema_50_4h_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.20
                position = 1
            # Short conditions: price breaks below Camarilla S1 AND 4h close < 4h EMA50 AND volume spike
            elif (close[i] < camarilla_s1_aligned[i] and 
                  close[i] < ema_50_4h_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price crosses below 4h EMA50 (trend reversal) OR price retouches Camarilla pivot (mean reversion)
            if close[i] < ema_50_4h_aligned[i] or abs(close[i] - camarilla_pp_aligned[i]) < 0.001 * camarilla_pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price crosses above 4h EMA50 (trend reversal) OR price retouches Camarilla pivot (mean reversion)
            if close[i] > ema_50_4h_aligned[i] or abs(close[i] - camarilla_pp_aligned[i]) < 0.001 * camarilla_pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals