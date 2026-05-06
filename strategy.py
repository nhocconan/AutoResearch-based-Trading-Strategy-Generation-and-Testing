#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R1/S1 breakout with 12h EMA50 trend filter and volume confirmation
# Uses Camarilla pivot levels from 4h timeframe for entry signals
# Requires price to break above R1 for long or below S1 for short
# Uses 12h EMA(50) to filter for trend direction (only trade in direction of higher timeframe trend)
# Volume confirmation (>1.5x 20-bar average) ensures participation
# Designed for 4h timeframe to target 75-200 total trades over 4 years (19-50/year)
# Works in both bull/bear: captures breakouts in trending markets, avoids false signals in consolidation

name = "4h_Camarilla_R1_S1_Breakout_12hEMA50_Trend_Volume_S"
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
    
    # Calculate HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_4h) < 20 or len(df_12h) < 50:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Camarilla pivot levels for 4h timeframe
    # Based on previous day's high, low, close
    # R1 = close + (high - low) * 1.1 / 12
    # S1 = close - (high - low) * 1.1 / 12
    # We use the previous 4h bar's high, low, close to calculate levels for current bar
    high_4h_prev = np.roll(high_4h, 1)
    low_4h_prev = np.roll(low_4h, 1)
    close_4h_prev = np.roll(close_4h, 1)
    # Set first value to NaN as there's no previous bar
    high_4h_prev[0] = np.nan
    low_4h_prev[0] = np.nan
    close_4h_prev[0] = np.nan
    
    # Calculate pivot levels
    R1 = close_4h_prev + (high_4h_prev - low_4h_prev) * 1.1 / 12
    S1 = close_4h_prev - (high_4h_prev - low_4h_prev) * 1.1 / 12
    
    # Calculate 12h EMA(50) trend filter
    close_12h = df_12h['close'].values
    ema_12h_50 = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate ATR(14) for 4h timeframe (for stoploss reference)
    tr1 = np.abs(high_4h[1:] - low_4h[1:])
    tr2 = np.abs(high_4h[1:] - close_4h[:-1])
    tr3 = np.abs(low_4h[1:] - close_4h[:-1])
    tr_4h = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_4h = pd.Series(tr_4h).rolling(window=14, min_periods=14).mean().values
    
    # Calculate volume confirmation filter (>1.5x 20-bar average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    # Align HTF indicators to 4h timeframe (primary)
    R1_aligned = align_htf_to_ltf(prices, df_4h, R1)
    S1_aligned = align_htf_to_ltf(prices, df_4h, S1)
    ema_12h_50_aligned = align_htf_to_ltf(prices, df_12h, ema_12h_50)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or np.isnan(ema_12h_50_aligned[i]) or 
            np.isnan(atr_4h[i]) or np.isnan(volume_filter[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: price breaks above R1 AND 12h EMA50 is rising (uptrend) AND volume confirmation
            if (close[i] > R1_aligned[i] and 
                ema_12h_50_aligned[i] > ema_12h_50_aligned[i-1] and  # EMA rising
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below S1 AND 12h EMA50 is falling (downtrend) AND volume confirmation
            elif (close[i] < S1_aligned[i] and 
                  ema_12h_50_aligned[i] < ema_12h_50_aligned[i-1] and  # EMA falling
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below S1 (reversal signal)
            if close[i] < S1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above R1 (reversal signal)
            if close[i] > R1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals