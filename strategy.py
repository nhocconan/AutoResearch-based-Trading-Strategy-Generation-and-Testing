#!/usr/bin/env python3
"""
Experiment #8331: 6-hour Camarilla Pivot with 1-day Trend Filter and Volume Confirmation.
Hypothesis: Camarilla pivot levels (S3/R3 for mean reversion, S4/R4 for breakout) combined with 
1-day EMA50 trend filter and volume confirmation (1.5x 20-period MA) provides high-probability 
entries. In bull markets, long at S3 bounce with trend; in bear markets, short at R3 rejection 
with trend. Breakouts at S4/R4 with volume and trend continuation capture strong moves. 
Targeting 100-200 total trades over 4 years for optimal balance of signal quality and cost.
"""

from mtf_data import get_htf_alias, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_8331_6h_camarilla1d_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
CAMARILLA_PERIOD = 1  # Use previous day's OHLC for pivot calculation
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
EMA_PERIOD = 50
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0
ATR_TARGET_MULTIPLIER = 3.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA for trend filter
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=EMA_PERIOD, adjust=False, min_periods=EMA_PERIOD).mean().values
    price_vs_ema = np.where(close_1d > ema_1d, 1, -1)  # 1=bullish, -1=bearish
    price_vs_ema_aligned = align_htf_to_ltf(prices, df_1d, price_vs_ema)
    
    # Calculate Camarilla levels from previous day's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point
    pivot = (high_1d + low_1d + close_1d) / 3.0
    # Camarilla levels
    s1 = close_1d - (high_1d - low_1d) * 1.0 / 12.0
    s2 = close_1d - (high_1d - low_1d) * 2.0 / 12.0
    s3 = close_1d - (high_1d - low_1d) * 3.0 / 12.0
    s4 = close_1d - (high_1d - low_1d) * 4.0 / 12.0
    r1 = close_1d + (high_1d - low_1d) * 1.0 / 12.0
    r2 = close_1d + (high_1d - low_1d) * 2.0 / 12.0
    r3 = close_1d + (high_1d - low_1d) * 3.0 / 12.0
    r4 = close_1d + (high_1d - low_1d) * 4.0 / 12.0
    
    # Align Camarilla levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    
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
    target_price = 0.0
    
    # Start from warmup period
    start = max(VOLUME_MA_PERIOD, ATR_PERIOD, EMA_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(price_vs_ema_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(r3_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Check stoploss or target
        if position == 1:  # long position
            if close[i] <= stop_price or close[i] >= target_price:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            if close[i] >= stop_price or close[i] <= target_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Determine market bias from 1d EMA
        bull_bias = price_vs_ema_aligned[i] == 1   # 1d close above EMA50
        bear_bias = price_vs_ema_aligned[i] == -1  # 1d close below EMA50
        
        # Volume confirmation
        volume_confirmed = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Mean reversion at S3/R3
        s3_bounce = bull_bias and (low[i] <= s3_aligned[i]) and (close[i] > s3_aligned[i])
        r3_rejection = bear_bias and (high[i] >= r3_aligned[i]) and (close[i] < r3_aligned[i])
        
        # Breakout continuation at S4/R4
        s4_breakout = bull_bias and (close[i] > s4_aligned[i-1]) and (i-1 >= 0) and not np.isnan(s4_aligned[i-1])
        r4_breakout = bear_bias and (close[i] < r4_aligned[i-1]) and (i-1 >= 0) and not np.isnan(r4_aligned[i-1])
        
        # Entry conditions
        long_entry = (s3_bounce or s4_breakout) and volume_confirmed
        short_entry = (r3_rejection or r4_breakout) and volume_confirmed
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
                target_price = entry_price + (ATR_TARGET_MULTIPLIER * atr[i])
            elif short_entry:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
                target_price = entry_price - (ATR_TARGET_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals
</s>