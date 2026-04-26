#!/usr/bin/env python3
"""
1d_Camarilla_R3_S3_Breakout_1wTrend_VolumeSpike_v1
Hypothesis: Daily Camarilla R3/S3 breakout with weekly trend alignment and volume spike confirmation. 
In trending markets (price above/below weekly EMA34), breakouts of Camarilla R3 (short) or S3 (long) 
are taken with volume > 1.5x 20-day average volume. In ranging markets, no trades are taken. 
Uses discrete position sizing (0.25) to minimize fee churn. Designed for low trade frequency 
(15-25/year) to avoid fee drag and work in both bull and bear regimes via trend filtering.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate ATR for stoploss reference (not used in signal generation directly)
    atr_period = 14
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum.reduce([tr1, tr2, tr3])
    atr = pd.Series(tr).ewm(span=atr_period, min_periods=atr_period, adjust=False).mean().values
    
    # Calculate weekly EMA34 for trend filter
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    htf_trend = np.where(close > ema_34_1w_aligned, 1, -1)  # 1 = uptrend, -1 = downtrend
    
    # Calculate Camarilla levels from previous day's OHLC
    # R3 = C + (H-L) * 1.1/2, S3 = C - (H-L) * 1.1/2
    # We need to shift by 1 to avoid look-ahead (use previous day's close)
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close[0] = close[0]  # first bar uses current close
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    
    camarilla_range = (prev_high - prev_low) * 1.1
    r3 = prev_close + camarilla_range / 2
    s3 = prev_close - camarilla_range / 2
    
    # Volume spike filter: volume > 1.5x 20-day average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 20 for volume MA, 1 for Camarilla)
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(r3[i]) or np.isnan(s3[i]) or 
            np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Only trade in alignment with weekly trend
        if htf_trend[i] == 1:  # Weekly uptrend - look for long breakouts
            if close[i] > s3[i] and volume_spike[i]:  # Price breaks above S3 with volume spike
                if position != 1:
                    signals[i] = 0.25
                    position = 1
                else:
                    signals[i] = 0.25
            elif close[i] < r3[i]:  # Price falls below R3 - exit long
                if position == 1:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.0  # stay flat
            else:
                # Hold current position
                if position == 0:
                    signals[i] = 0.0
                elif position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25  # should not happen in uptrend
        elif htf_trend[i] == -1:  # Weekly downtrend - look for short breakouts
            if close[i] < r3[i] and volume_spike[i]:  # Price breaks below R3 with volume spike
                if position != -1:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = -0.25
            elif close[i] > s3[i]:  # Price rises above S3 - exit short
                if position == -1:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.0  # stay flat
            else:
                # Hold current position
                if position == 0:
                    signals[i] = 0.0
                elif position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals

name = "1d_Camarilla_R3_S3_Breakout_1wTrend_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0