#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_ATRstop_v2
Hypothesis: 4-hour Camarilla R1/S1 breakout with 1-day trend filter (price > 1d EMA50) and volume confirmation (>1.8x 20-period average).
Long when price breaks above R1 in 1-day uptrend with volume confirmation.
Short when price breaks below S1 in 1-day downtrend with volume confirmation.
Exit via ATR trailing stop (2.0*ATR from extreme) or opposite Camarilla level (S1 for long, R1 for short).
Camarilla levels provide precise intraday support/resistance derived from prior day's range.
Volume confirmation ensures breakouts have conviction. 1-day trend filter aligns with higher timeframe bias.
Designed for ~75-200 trades over 4 years (19-50/year) via tight Camarilla breakout conditions.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla calculation and trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:  # need at least 1 day for prior day's OHLC
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    open_1d = df_1d['open'].values
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Camarilla levels for 1d (based on prior day's OHLC)
    # Camarilla R1 = close + (high - low) * 1.1/12
    # Camarilla S1 = close - (high - low) * 1.1/12
    # Using prior day's values (shifted by 1)
    prior_close = np.roll(close_1d, 1)
    prior_high = np.roll(high_1d, 1)
    prior_low = np.roll(low_1d, 1)
    # First day has no prior day, so set to NaN
    prior_close[0] = np.nan
    prior_high[0] = np.nan
    prior_low[0] = np.nan
    
    camarilla_range = prior_high - prior_low
    r1 = prior_close + camarilla_range * 1.1 / 12
    s1 = prior_close - camarilla_range * 1.1 / 12
    
    # Align Camarilla levels to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # ATR for stoploss (14-period)
    atr_period = 14
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    # Volume regime: volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_regime = volume > (1.8 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    long_extreme = 0.0   # highest close since long entry
    short_extreme = 0.0  # lowest close since short entry
    
    # Start index: need warmup for calculations
    start_idx = max(100, atr_period, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        ema_trend = ema_50_1d_aligned[i]
        r1 = r1_aligned[i]
        s1 = s1_aligned[i]
        
        if position == 0:
            # Only trade in trending regimes (1d EMA50 filter)
            if close[i] > ema_trend:  # 1d uptrend regime
                # Long: break above R1 with volume confirmation
                long_signal = (close[i] > r1) and vol_regime[i]
            else:  # 1d downtrend regime
                # Short: break below S1 with volume confirmation
                short_signal = (close[i] < s1) and vol_regime[i]
            
            if 'long_signal' in locals() and long_signal:
                signals[i] = 0.25
                position = 1
                long_extreme = close[i]
            elif 'short_signal' in locals() and short_signal:
                signals[i] = -0.25
                position = -1
                short_extreme = close[i]
            else:
                signals[i] = 0.0
                # Clear signal variables for next iteration
                if 'long_signal' in locals(): del long_signal
                if 'short_signal' in locals(): del short_signal
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Update extreme for trailing stop
            if close[i] > long_extreme:
                long_extreme = close[i]
            # Exit conditions: 
            # 1. ATR trailing stop (2.0*ATR from extreme)
            atr_stop = long_extreme - 2.0 * atr[i]
            # 2. Price breaks below S1 (opposite Camarilla level)
            if close[i] <= atr_stop or close[i] < s1:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Update extreme for trailing stop
            if close[i] < short_extreme:
                short_extreme = close[i]
            # Exit conditions:
            # 1. ATR trailing stop (2.0*ATR from extreme)
            atr_stop = short_extreme + 2.0 * atr[i]
            # 2. Price breaks above R1 (opposite Camarilla level)
            if close[i] >= atr_stop or close[i] > r1:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_ATRstop_v2"
timeframe = "4h"
leverage = 1.0