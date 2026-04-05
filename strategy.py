#!/usr/bin/env python3
"""
Experiment #7711: 6-hour Camarilla Pivot Reversal with Daily Trend Filter
Hypothesis: Price reversing from Camarilla R3/S3 levels on 6h with daily trend alignment
captures mean-reversion in ranging markets and breakout continuation in trending markets.
Works in bull markets (long at S3/S4, short at R3/R4) and bear markets (reverse logic).
Targets 50-150 trades over 4 years (12-37/year).
"""

from mtf_data import get_athf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7711_6h_camarilla1d_trend_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
CAMARILLA_PERIOD = 1  # Use previous day's OHLC
EMA_TREND = 50
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.3
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

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
    
    # Calculate daily OHLC for Camarilla (use previous day's values)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels from previous day's OHLC
    # R4 = Close + 1.5*(High-Low), R3 = Close + 1.1*(High-Low)
    # S3 = Close - 1.1*(High-Low), S4 = Close - 1.5*(High-Low)
    camarilla_range = high_1d - low_1d
    r4 = close_1d + 1.5 * camarilla_range
    r3 = close_1d + 1.1 * camarilla_range
    s3 = close_1d - 1.1 * camarilla_range
    s4 = close_1d - 1.5 * camarilla_range
    
    # Align Camarilla levels to 6h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
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
        
        # Determine market regime
        bull_regime = close[i] > ema_1d_aligned[i]   # price above 1d EMA
        bear_regime = close[i] < ema_1d_aligned[i]   # price below 1d EMA
        
        # Volume confirmation
        volume_confirmed = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Camarilla reversal conditions
        # Long when price touches/bounces from S3/S4 in bull regime, or breaks S4 in bear regime
        long_signal = False
        short_signal = False
        
        if bull_regime:
            # In bull trend: mean reversion at S3, breakout continuation above R4
            long_signal = (low[i] <= s3_aligned[i] and close[i] > s3_aligned[i]) and volume_confirmed
            short_signal = (high[i] >= r4_aligned[i] and close[i] < r4_aligned[i]) and volume_confirmed
        else:  # bear regime
            # In bear trend: mean reversion at R3, breakdown continuation below S4
            long_signal = (low[i] <= s4_aligned[i] and close[i] > s4_aligned[i]) and volume_confirmed
            short_signal = (high[i] >= r3_aligned[i] and close[i] < r3_aligned[i]) and volume_confirmed
        
        # Generate signals
        if position == 0:
            if long_signal:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif short_signal:
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