#!/usr/bin/env python3
"""
1d_Camarilla_R3S3_Breakout_1wTrend_VolumeSpike_ATRstop
Hypothesis: Daily Camarilla R3/S3 level breakout with 1-week EMA50 trend filter and volume confirmation (>2.0x 20-period average).
Long when price breaks above R3 in 1w uptrend with volume spike. Short when price breaks below S3 in 1w downtrend with volume spike.
Exit via ATR trailing stop (2.5*ATR from extreme) or opposite Camarilla level (S3 for longs, R3 for shorts).
Camarilla levels provide mathematically derived support/resistance that work well in both trending and ranging markets.
Weekly trend filter ensures alignment with higher timeframe momentum. Volume spike confirms breakout conviction.
Designed for ~30-80 trades over 4 years (7-20/year) via tight Camarilla breakout conditions.
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
    
    # Get 1w data for trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:  # need 50 for EMA50
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate 1w EMA50 for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
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
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        ema_trend = ema_50_1w_aligned[i]
        
        if position == 0:
            # Calculate Camarilla levels for today using previous day's OHLC
            if i >= 1:  # need previous day's data
                # Get previous day's OHLC (assuming daily data aligned to 1d timeframe)
                # Since we're on 1d timeframe, we can use daily OHLC directly
                prev_high = high[i-1]
                prev_low = low[i-1]
                prev_close = close[i-1]
                
                # Camarilla levels
                range_val = prev_high - prev_low
                r3 = prev_close + (range_val * 1.1 / 4)
                s3 = prev_close - (range_val * 1.1 / 4)
                
                # Only trade in trending regimes (1w EMA50 filter)
                if close[i] > ema_trend:  # 1w uptrend regime
                    # Long: break above R3 with volume confirmation
                    long_signal = (close[i] > r3) and vol_regime[i]
                else:  # 1w downtrend regime
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
            else:
                signals[i] = 0.0
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
            if i >= 1:
                prev_high = high[i-1]
                prev_low = low[i-1]
                prev_close = close[i-1]
                range_val = prev_high - prev_low
                s3 = prev_close - (range_val * 1.1 / 4)
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
            if i >= 1:
                prev_high = high[i-1]
                prev_low = low[i-1]
                prev_close = close[i-1]
                range_val = prev_high - prev_low
                r3 = prev_close + (range_val * 1.1 / 4)
                if close[i] >= atr_stop or close[i] > r3:
                    signals[i] = 0.0
                    position = 0
    
    return signals

name = "1d_Camarilla_R3S3_Breakout_1wTrend_VolumeSpike_ATRstop"
timeframe = "1d"
leverage = 1.0