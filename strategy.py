#!/usr/bin/env python3
"""
6h_Camarilla_R4S4_Breakout_1dTrend_VolumeSpike
Hypothesis: 6h Camarilla R4/S4 breakout with 1d EMA50 trend filter and volume confirmation.
Long when price breaks above R4 in 1d uptrend (close > 1d EMA50) with volume > 2.0x 20-period average.
Short when price breaks below S4 in 1d downtrend (close < 1d EMA50) with volume > 2.0x 20-period average.
Uses 1d HTF for trend alignment and volume spike for confirmation to avoid false breakouts.
Designed for ~12-25 trades/year by requiring strong breakouts at extreme Camarilla levels with volume confirmation.
Works in bull/bear markets via 1d EMA50 filter; avoids whipsaws via volume confirmation.
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
    
    # Get 1d data for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Camarilla levels for 6h timeframe using previous day's OHLC
    # Camarilla: based on previous day's range
    # R4 = close + 1.5*(high - low)
    # S4 = close - 1.5*(high - low)
    prev_close = np.roll(close_1d, 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close[0] = close_1d[0]  # first period
    prev_high[0] = high_1d[0]
    prev_low[0] = low_1d[0]
    
    camarilla_range = prev_high - prev_low
    r4 = prev_close + 1.5 * camarilla_range
    s4 = prev_close - 1.5 * camarilla_range
    
    # Align Camarilla levels to 6h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Volume regime: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_regime = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for calculations
    start_idx = max(100, 50)  # EMA50 needs 50 periods
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(r4_aligned[i]) or 
            np.isnan(s4_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        ema_trend = ema_50_1d_aligned[i]
        r4_level = r4_aligned[i]
        s4_level = s4_aligned[i]
        
        if position == 0:
            # Only trade in trending regimes (1d EMA50 filter)
            if close[i] > ema_trend:  # 1d uptrend regime
                # Long: break above R4 with volume spike
                long_signal = (close[i] > r4_level) and vol_regime[i]
            else:  # 1d downtrend regime
                # Short: break below S4 with volume spike
                short_signal = (close[i] < s4_level) and vol_regime[i]
            
            if 'long_signal' in locals() and long_signal:
                signals[i] = 0.25
                position = 1
            elif 'short_signal' in locals() and short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
                # Clear signal variables for next iteration
                if 'long_signal' in locals(): del long_signal
                if 'short_signal' in locals(): del short_signal
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: price re-enters below R4 (breakout failed) or reverses below S4
            if close[i] < r4_level:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price re-enters above S4 (breakout failed) or reverses above R4
            if close[i] > s4_level:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Camarilla_R4S4_Breakout_1dTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0