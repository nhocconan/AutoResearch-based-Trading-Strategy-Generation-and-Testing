#!/usr/bin/env python3
name = "12h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0

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
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate daily Camarilla levels from previous day (complete day only)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Use previous day's complete data to calculate today's Camarilla
    prev_high = high_1d[:-1]
    prev_low = low_1d[:-1]
    prev_close = close_1d[:-1]
    
    # Need at least one complete day
    if len(prev_high) < 1:
        return np.zeros(n)
    
    # Calculate previous day's Camarilla levels (R3, S3, R4, S4)
    # Camarilla formulas: 
    # R4 = close + (high - low) * 1.1 / 2
    # R3 = close + (high - low) * 1.1 / 4
    # S3 = close - (high - low) * 1.1 / 4
    # S4 = close - (high - low) * 1.1 / 2
    hl_range = prev_high - prev_low
    r3 = prev_close + hl_range * 1.1 / 4
    s3 = prev_close - hl_range * 1.1 / 4
    r4 = prev_close + hl_range * 1.1 / 2
    s4 = prev_close - hl_range * 1.1 / 2
    
    # Create arrays for each day (align with days)
    r3_per_day = np.full(len(df_1d), np.nan)
    s3_per_day = np.full(len(df_1d), np.nan)
    r4_per_day = np.full(len(df_1d), np.nan)
    s4_per_day = np.full(len(df_1d), np.nan)
    
    # Shift by one day: current day gets previous day's levels
    r3_per_day[1:] = r3
    s3_per_day[1:] = s3
    r4_per_day[1:] = r4
    s4_per_day[1:] = s4
    
    # Align to 12h timeframe (only complete daily levels available)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_per_day)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_per_day)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4_per_day)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4_per_day)
    
    # Calculate daily EMA(34) for trend filter
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Volume spike detection (20-period average on 12h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for volatility filter (14-period)
    high_low = high - low
    high_close = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    low_close = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 14)  # Wait for volume MA and ATR
    
    for i in range(start_idx, n):
        if np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(r4_aligned[i]) or \
           np.isnan(s4_aligned[i]) or np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price > R3, above daily EMA34, volume spike, not extreme volatility
            vol_condition = volume[i] > vol_ma[i] * 1.5
            vol_not_extreme = atr[i] < np.median(atr[max(0, i-50):i+1]) * 3
            
            if (close[i] > r3_aligned[i] and 
                close[i] > ema_34_aligned[i] and 
                vol_condition and 
                vol_not_extreme):
                signals[i] = 0.30
                position = 1
            # Short: price < S3, below daily EMA34, volume spike, not extreme volatility
            elif (close[i] < s3_aligned[i] and 
                  close[i] < ema_34_aligned[i] and 
                  vol_condition and 
                  vol_not_extreme):
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit: price < S3 or below EMA34 or volatility spike
            if (close[i] < s3_aligned[i] or 
                close[i] < ema_34_aligned[i] or
                atr[i] > np.median(atr[max(0, i-50):i+1]) * 4):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit: price > R3 or above EMA34 or volatility spike
            if (close[i] > r3_aligned[i] or 
                close[i] > ema_34_aligned[i] or
                atr[i] > np.median(atr[max(0, i-50):i+1]) * 4):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

# Hypothesis: 12h Camarilla R3/S3 breakout with daily trend filter and volume confirmation.
# Uses previous day's Camarilla levels (R3, S3) as key support/resistance.
# Breakout above R3 with volume suggests bullish momentum; breakdown below S3 suggests bearish.
# Daily EMA(34) ensures we trade only in direction of daily trend.
# Volume confirmation ensures institutional participation.
# Volatility filter avoids whipsaws during extreme volatility spikes.
# Position size 0.30 balances risk and keeps trade frequency ~15-30 trades/year on 12h timeframe.
# Works in bull markets (buy breakouts above R3 in uptrend) and bear markets (sell breakdowns below S3 in downtrend).