#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeFilter
Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA200 trend filter and volume confirmation.
Long when price breaks above R1 in 4h uptrend (close > 4h EMA200) with volume > 1.8x 24-bar average.
Short when price breaks below S1 in 4h downtrend (close < 4h EMA200) with volume > 1.8x 24-bar average.
Exit via ATR-based trailing stop (2.5*ATR from extreme) or re-entry into H3-L3 range.
Designed for moderate trade frequency (~20-40/year) by requiring R1/S1 breakout + volume filter + 4h trend.
Uses 1h timeframe for entry timing precision while deriving signal direction from 4h HTF.
Works in bull/bear markets via 4h EMA200 filter; avoids whipsaws in ranging markets via volume confirmation.
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
    
    # Get 4h data for Camarilla pivot calculation and trend filter (HTF)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Camarilla levels for previous 4h bar
    prev_close = np.concatenate([[np.nan], close_4h[:-1]])
    prev_high = np.concatenate([[np.nan], high_4h[:-1]])
    prev_low = np.concatenate([[np.nan], low_4h[:-1]])
    
    pivot = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    r1 = prev_close + range_hl * 1.1 / 12
    s1 = prev_close - range_hl * 1.1 / 12
    h3 = prev_close + range_hl * 1.1 / 4
    l3 = prev_close - range_hl * 1.1 / 4
    
    # Align Camarilla levels to 1h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_4h, r1)
    s1_aligned = align_htf_to_ltf(prices, df_4h, s1)
    h3_aligned = align_htf_to_ltf(prices, df_4h, h3)
    l3_aligned = align_htf_to_ltf(prices, df_4h, l3)
    
    # Get 4h data for trend filter (EMA200)
    ema_200_4h = pd.Series(close_4h).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_200_4h)
    
    # Volume regime: volume > 1.8x 24-period average (1h timeframe)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_regime = volume > (1.8 * vol_ma_24)
    
    # ATR for trailing stop (14-period)
    atr_period = 14
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    long_high = 0.0   # highest close since long entry
    short_low = 0.0   # lowest close since short entry
    
    # Start index: need warmup for calculations
    start_idx = max(100, atr_period)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or 
            np.isnan(ema_200_4h_aligned[i]) or np.isnan(vol_ma_24[i]) or 
            np.isnan(atr[i])):
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        ema_trend = ema_200_4h_aligned[i]
        
        if position == 0:
            # Only trade in trending regimes (4h EMA200 filter)
            if close[i] > ema_trend:  # 4h uptrend regime
                # Long: break above R1 with volume confirmation
                long_signal = (close[i] > r1_aligned[i]) and vol_regime[i]
            else:  # 4h downtrend regime
                # Short: break below S1 with volume confirmation
                short_signal = (close[i] < s1_aligned[i]) and vol_regime[i]
            
            if 'long_signal' in locals() and long_signal:
                signals[i] = 0.20
                position = 1
                long_high = close[i]
            elif 'short_signal' in locals() and short_signal:
                signals[i] = -0.20
                position = -1
                short_low = close[i]
            else:
                signals[i] = 0.0
                # Clear signal variables for next iteration
                if 'long_signal' in locals(): del long_signal
                if 'short_signal' in locals(): del short_signal
        elif position == 1:
            # Long: hold position
            signals[i] = 0.20
            # Update highest close
            if close[i] > long_high:
                long_high = close[i]
            # Exit conditions: ATR trailing stop OR re-enter H3-L3 range
            atr_stop = long_high - 2.5 * atr[i]
            range_exit = (close[i] < h3_aligned[i] and close[i] > l3_aligned[i])
            if close[i] <= atr_stop or range_exit:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.20
            # Update lowest close
            if close[i] < short_low:
                short_low = close[i]
            # Exit conditions: ATR trailing stop OR re-enter H3-L3 range
            atr_stop = short_low + 2.5 * atr[i]
            range_exit = (close[i] > l3_aligned[i] and close[i] < h3_aligned[i])
            if close[i] >= atr_stop or range_exit:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeFilter"
timeframe = "1h"
leverage = 1.0