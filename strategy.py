#!/usr/bin/env python3
"""
1h_Camarilla_R3_S3_Breakout_4hTrend_VolumeConfirmation
Hypothesis: 1-hour Camarilla R3/S3 breakout with 4-hour trend filter (price > 4h EMA34) and volume confirmation (>1.5x 20-period average).
Long when price breaks above R3 in 4h uptrend with volume confirmation.
Short when price breaks below S3 in 4h downtrend with volume confirmation.
Exit via opposite Camarilla level (S3 for longs, R3 for shorts) or ATR trailing stop (1.5*ATR from extreme).
Designed for 60-150 total trades over 4 years (15-37/year) via tight Camarilla breakout conditions.
Uses 4h for signal direction, 1h only for entry timing. Session filter (08-20 UTC) reduces noise.
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
    
    # Get 4h data for trend filter and Camarilla calculation (HTF)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 40:  # need sufficient data for calculations
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate 4h EMA34 for trend filter
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Calculate Camarilla levels on 4h data (using previous day's OHLC)
    # Camarilla equations based on previous 4h bar's range
    # R4 = close + 1.5*(high-low), R3 = close + 1.25*(high-low), etc.
    # But for intraday, we use daily OHLC to calculate levels
    # Since we're on 1h chart, we need daily OHLC for Camarilla
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Use previous day's OHLC for today's Camarilla levels
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Calculate Camarilla levels
    R3 = prev_close + 1.1 * (prev_high - prev_low) * 1.1 / 2  # R3 = C + 1.1*(H-L)*1.1/2
    S3 = prev_close - 1.1 * (prev_high - prev_low) * 1.1 / 2  # S3 = C - 1.1*(H-L)*1.1/2
    R4 = prev_close + 1.1 * (prev_high - prev_low) * 1.1 / 2 * 2/1.1  # R4 = C + 1.1*(H-L)
    S4 = prev_close - 1.1 * (prev_high - prev_low) * 1.1 / 2 * 2/1.1  # S4 = C - 1.1*(H-L)
    
    # Simplified: R3 = close + 1.1*(high-low), S3 = close - 1.1*(high-low)
    # Actually standard Camarilla: R3 = close + 1.1*(high-low), S3 = close - 1.1*(high-low)
    R3 = prev_close + 1.1 * (prev_high - prev_low)
    S3 = prev_close - 1.1 * (prev_high - prev_low)
    R4 = prev_close + 1.5 * (prev_high - prev_low)
    S4 = prev_close - 1.5 * (prev_high - prev_low)
    
    # Align Camarilla levels to 1h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    R4_aligned = align_htf_to_ltf(prices, df_1d, R4)
    S4_aligned = align_htf_to_ltf(prices, df_1d, S4)
    
    # ATR for stoploss (20-period)
    atr_period = 20
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    # Volume regime: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_regime = volume > (1.5 * vol_ma_20)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    long_extreme = 0.0   # highest close since long entry
    short_extreme = 0.0  # lowest close since short entry
    
    # Start index: need warmup for calculations
    start_idx = max(100, atr_period, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready or outside session
        if (np.isnan(ema_34_4h_aligned[i]) or np.isnan(R3_aligned[i]) or 
            np.isnan(S3_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma_20[i]) or
            not in_session[i]):
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        ema_trend = ema_34_4h_aligned[i]
        R3 = R3_aligned[i]
        S3 = S3_aligned[i]
        R4 = R4_aligned[i]
        S4 = S4_aligned[i]
        
        if position == 0:
            # Only trade in trending regimes (4h EMA34 filter)
            if close[i] > ema_trend:  # 4h uptrend regime
                # Long: break above R3 with volume confirmation
                long_signal = (close[i] > R3) and vol_regime[i]
            else:  # 4h downtrend regime
                # Short: break below S3 with volume confirmation
                short_signal = (close[i] < S3) and vol_regime[i]
            
            if 'long_signal' in locals() and long_signal:
                signals[i] = 0.20
                position = 1
                long_extreme = close[i]
            elif 'short_signal' in locals() and short_signal:
                signals[i] = -0.20
                position = -1
                short_extreme = close[i]
            else:
                signals[i] = 0.0
                # Clear signal variables for next iteration
                if 'long_signal' in locals(): del long_signal
                if 'short_signal' in locals(): del short_signal
        elif position == 1:
            # Long: hold position
            signals[i] = 0.20
            # Update extreme for trailing stop
            if close[i] > long_extreme:
                long_extreme = close[i]
            # Exit conditions: 
            # 1. ATR trailing stop (1.5*ATR from extreme)
            atr_stop = long_extreme - 1.5 * atr[i]
            # 2. Price breaks below S3 (opposite Camarilla level)
            if close[i] <= atr_stop or close[i] < S3:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.20
            # Update extreme for trailing stop
            if close[i] < short_extreme:
                short_extreme = close[i]
            # Exit conditions:
            # 1. ATR trailing stop (1.5*ATR from extreme)
            atr_stop = short_extreme + 1.5 * atr[i]
            # 2. Price breaks above R3 (opposite Camarilla level)
            if close[i] >= atr_stop or close[i] > R3:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_Camarilla_R3_S3_Breakout_4hTrend_VolumeConfirmation"
timeframe = "1h"
leverage = 1.0