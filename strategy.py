#!/usr/bin/env python3
"""
Experiment #7791: 6-hour Camarilla pivot reversal with 1-day trend filter and volume confirmation.
Hypothesis: In ranging markets (common in 2025-2026), price tends to revert from extreme Camarilla levels (R3/S3, R4/S4). 
When price reaches R3/S3 in a 1-day uptrend/downtrend, we take counter-trend positions with volume confirmation.
In strong trends (price beyond R4/S4), we follow the breakout. This adapts to both trending and ranging regimes.
"""

from mtf_data import get_hrf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7791_6h_camarilla_pivot_reversal_1d_trend_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
CAMARILLA_PERIOD = 1
EMA_TREND = 50
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA for trend filter
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=EMA_TREND, adjust=False, min_periods=EMA_TREND).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate Camarilla levels from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_prev = df_1d['close'].values
    
    # Camarilla levels: based on previous day's range
    range_1d = high_1d - low_1d
    # R3, R4, S3, S4 levels
    r3 = close_1d_prev + range_1d * 1.1 / 2
    r4 = close_1d_prev + range_1d * 1.1
    s3 = close_1d_prev - range_1d * 1.1 / 2
    s4 = close_1d_prev - range_1d * 1.1
    
    # Align Camarilla levels to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume moving average
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR for risk management
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(EMA_TREND, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(ema_1d_aligned[i]) or np.isnan(r3_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Determine market regime from 1d EMA
        bull_regime = close[i] > ema_1d_aligned[i]   # price above 1d EMA = uptrend
        bear_regime = close[i] < ema_1d_aligned[i]   # price below 1d EMA = downtrend
        
        # Volume confirmation
        volume_confirmed = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Camarilla level conditions
        # In uptrend: look for reversals at S3/S4, breakouts at R3/R4
        # In downtrend: look for reversals at R3/R4, breakouts at S3/S4
        near_s3 = abs(close[i] - s3_aligned[i]) < (range_1d[i] * 0.05) if not np.isnan(range_1d[i]) else False
        near_s4 = abs(close[i] - s4_aligned[i]) < (range_1d[i] * 0.05) if not np.isnan(range_1d[i]) else False
        near_r3 = abs(close[i] - r3_aligned[i]) < (range_1d[i] * 0.05) if not np.isnan(range_1d[i]) else False
        near_r4 = abs(close[i] - r4_aligned[i]) < (range_1d[i] * 0.05) if not np.isnan(range_1d[i]) else False
        
        # Breakout conditions (price beyond R4/S4)
        breakout_up = close[i] > r4_aligned[i]
        breakout_down = close[i] < s4_aligned[i]
        
        # Entry logic
        long_entry = False
        short_entry = False
        
        if bull_regime:
            # In uptrend: mean revert from S3/S4, breakout through R3/R4
            if (near_s3 or near_s4) and volume_confirmed:
                long_entry = True  # bounce from support
            elif breakout_up and volume_confirmed:
                long_entry = True  # breakout continuation
        elif bear_regime:
            # In downtrend: mean revert from R3/R4, breakout through S3/S4
            if (near_r3 or near_r4) and volume_confirmed:
                short_entry = True  # rejection from resistance
            elif breakout_down and volume_confirmed:
                short_entry = True  # breakdown continuation
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif short_entry:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals