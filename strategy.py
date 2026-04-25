#!/usr/bin/env python3
"""
4h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike_ATRstop_v3
Hypothesis: 4-hour Camarilla R3/S3 breakout with 1-day EMA50 trend filter and volume confirmation (>2.0x 20-period average).
Long when price breaks above R3 in 1-day uptrend with volume confirmation.
Short when price breaks below S3 in 1-day downtrend with volume confirmation.
Exit via ATR trailing stop (2.5*ATR from extreme) or opposite Camarilla level (S3 for longs, R3 for shorts).
Camarilla levels provide mathematically derived support/resistance that works well in both trending and ranging markets.
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
    
    # Get 1d data for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # need 50 for EMA
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate ATR for stoploss (14-period)
    atr_period = 14
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    # Volume regime: volume > 2.0x 20-period average (stricter to reduce trades)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_regime = volume > (2.0 * vol_ma_20)
    
    # Calculate Camarilla levels for 4h timeframe using daily OHLC from previous day
    # Camarilla levels: based on previous day's range
    # We need to get daily OHLC and align to 4h bars
    df_1d_ohlc = get_htf_data(prices, '1d')  # already have df_1d
    if len(df_1d_ohlc) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC (shifted by 1 to avoid look-ahead)
    prev_close = df_1d_ohlc['close'].shift(1).values
    prev_high = df_1d_ohlc['high'].shift(1).values
    prev_low = df_1d_ohlc['low'].shift(1).values
    prev_open = df_1d_ohlc['open'].shift(1).values
    
    # Typical price for Camarilla calculation
    typical_price = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    
    # Camarilla levels
    R3 = typical_price + (range_hl * 1.1 / 4)
    R2 = typical_price + (range_hl * 1.1 / 6)
    R1 = typical_price + (range_hl * 1.1 / 12)
    S1 = typical_price - (range_hl * 1.1 / 12)
    S2 = typical_price - (range_hl * 1.1 / 6)
    S3 = typical_price - (range_hl * 1.1 / 4)
    
    # Align Camarilla levels to 4h timeframe (previous day's levels are valid for current day)
    R3_aligned = align_htf_to_ltf(prices, df_1d_ohlc, R3)
    R2_aligned = align_htf_to_ltf(prices, df_1d_ohlc, R2)
    R1_aligned = align_htf_to_ltf(prices, df_1d_ohlc, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d_ohlc, S1)
    S2_aligned = align_htf_to_ltf(prices, df_1d_ohlc, S2)
    S3_aligned = align_htf_to_ltf(prices, df_1d_ohlc, S3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    long_extreme = 0.0   # highest close since long entry
    short_extreme = 0.0  # lowest close since short entry
    
    # Start index: need warmup for calculations
    start_idx = max(100, atr_period, 20, 50)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma_20[i]) or
            np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        ema_trend = ema_50_1d_aligned[i]
        r3 = R3_aligned[i]
        s3 = S3_aligned[i]
        
        if position == 0:
            # Only trade in trending regimes (1d EMA50 filter)
            if close[i] > ema_trend:  # 1d uptrend regime
                # Long: break above R3 with volume confirmation
                long_signal = (close[i] > r3) and vol_regime[i]
            else:  # 1d downtrend regime
                # Short: break below S3 with volume confirmation
                short_signal = (close[i] < s3) and vol_regime[i]
            
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
            # 2. Price breaks below S3 (opposite Camarilla level)
            if close[i] <= atr_stop or close[i] < s3:
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
            # 2. Price breaks above R3 (opposite Camarilla level)
            if close[i] >= atr_stop or close[i] > r3:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike_ATRstop_v3"
timeframe = "4h"
leverage = 1.0