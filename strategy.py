#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_12hTrend_VolumeSpike
Hypothesis: 4-hour Camarilla R1/S1 breakout with 12-hour EMA50 trend filter and volume confirmation (>2.0x 20-period average).
Long when price breaks above R1 in 12h uptrend with volume confirmation.
Short when price breaks below S1 in 12h downtrend with volume confirmation.
Exit via opposite Camarilla level (S1 for longs, R1 for shorts) or ATR trailing stop (2.5*ATR from extreme).
Camarilla levels provide precise intraday support/resistance that works well in ranging and trending markets.
Volume confirmation ensures breakouts have conviction. 12h trend filter aligns with higher timeframe bias.
Designed for ~75-150 trades over 4 years (19-38/year) via tight Camarilla breakout conditions.
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
    
    # Get 12h data for trend filter and Camarilla calculation (HTF)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:  # need sufficient data for EMA
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h_series = pd.Series(close_12h)
    
    # Calculate 12h EMA50 for trend filter
    ema_50_12h = close_12h_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Camarilla levels from previous 12h bar
    # Camarilla: based on previous day's (here: previous 12h bar) range
    # R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low), 
    # R2 = close + 0.55*(high-low), R1 = close + 0.275*(high-low)
    # PP = (high+low+close)/3
    # S1 = close - 0.275*(high-low), S2 = close - 0.55*(high-low)
    # S3 = close - 1.1*(high-low), S4 = close - 1.5*(high-low)
    
    # Shift to get previous bar's OHLC
    prev_high = np.roll(high_12h, 1)
    prev_low = np.roll(low_12h, 1)
    prev_close = np.roll(close_12h, 1)
    prev_high[0] = high_12h[0]  # first bar
    prev_low[0] = low_12h[0]
    prev_close[0] = close_12h[0]
    
    # Calculate Camarilla levels for each 12h bar
    camarilla_r1 = prev_close + 0.275 * (prev_high - prev_low)
    camarilla_s1 = prev_close - 0.275 * (prev_high - prev_low)
    camarilla_r2 = prev_close + 0.55 * (prev_high - prev_low)
    camarilla_s2 = prev_close - 0.55 * (prev_high - prev_low)
    
    # Align Camarilla levels to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r1)
    s1_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s1)
    r2_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r2)
    s2_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s2)
    
    # ATR for stoploss (20-period)
    atr_period = 20
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    # Volume regime: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_regime = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    long_extreme = 0.0   # highest close since long entry
    short_extreme = 0.0  # lowest close since short entry
    
    # Start index: need warmup for calculations
    start_idx = max(100, atr_period, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        ema_trend = ema_50_12h_aligned[i]
        r1 = r1_aligned[i]
        s1 = s1_aligned[i]
        r2 = r2_aligned[i]
        s2 = s2_aligned[i]
        
        if position == 0:
            # Only trade in trending regimes (12h EMA50 filter)
            if close[i] > ema_trend:  # 12h uptrend regime
                # Long: break above R1 with volume confirmation
                long_signal = (close[i] > r1) and vol_regime[i]
            else:  # 12h downtrend regime
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
            # 1. ATR trailing stop (2.5*ATR from extreme)
            atr_stop = long_extreme - 2.5 * atr[i]
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
            # 1. ATR trailing stop (2.5*ATR from extreme)
            atr_stop = short_extreme + 2.5 * atr[i]
            # 2. Price breaks above R1 (opposite Camarilla level)
            if close[i] >= atr_stop or close[i] > r1:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_12hTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0