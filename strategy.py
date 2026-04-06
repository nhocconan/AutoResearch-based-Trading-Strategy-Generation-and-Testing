#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot levels with 1d RSI filter and volume confirmation.
# In ranging markets, price tends to revert from R3/S3 levels; in trending markets,
# breakouts beyond R4/S4 continue. The 1d RSI filter ensures we only take trades
# aligned with higher timeframe momentum, reducing false signals. Volume confirms
# institutional participation. This should work in both bull and bear markets by
# adapting to regime via price action at key levels.
# Target: 50-150 total trades over 4 years (12-37/year).

name = "exp_13067_6h_camarilla1d_rsi_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
CAMARILLA_PERIOD = 1   # Use previous day's OHLC
RSI_PERIOD = 14
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_rsi(close, period):
    """Calculate RSI using Wilder's smoothing"""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily RSI for trend filter
    close_1d = df_1d['close'].values
    rsi_1d = calculate_rsi(close_1d, RSI_PERIOD)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Calculate previous day's OHLC for Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: based on previous day's range
    # R4 = close + 1.5 * (high - low)
    # R3 = close + 1.1 * (high - low)
    # S3 = close - 1.1 * (high - low)
    # S4 = close - 1.5 * (high - low)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = np.nan  # First day has no previous
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    camarilla_range = prev_high - prev_low
    r4 = prev_close + 1.1 * camarilla_range  # Actually 1.1 for R3, 1.5 for R4 - fixing
    r3 = prev_close + 1.1 * camarilla_range
    s3 = prev_close - 1.1 * camarilla_range
    s4 = prev_close - 1.5 * camarilla_range
    
    # Correct Camarilla formulas
    r3 = prev_close + 1.1 * camarilla_range
    r4 = prev_close + 1.5 * camarilla_range
    s3 = prev_close - 1.1 * camarilla_range
    s4 = prev_close - 1.5 * camarilla_range
    
    # Align Camarilla levels to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # ATR for stoploss
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(RSI_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if RSI not available
        if np.isnan(rsi_1d_aligned[i]):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
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
        
        # Volume confirmation (20-period MA)
        if i < VOLUME_MA_PERIOD:
            volume_ok = False
        else:
            volume_ma = np.mean(volume[i-VOLUME_MA_PERIOD:i])
            volume_ok = volume[i] > (volume_ma * VOLUME_THRESHOLD)
        
        # RSI filter: avoid overbought/oversold extremes
        rsi_val = rsi_1d_aligned[i]
        rsi_overbought = rsi_val > 70
        rsi_oversold = rsi_val < 30
        # Only take longs when not overbought, shorts when not oversold
        rsi_long_ok = not rsi_overbought
        rsi_short_ok = not rsi_oversold
        
        # Camarilla-based signals
        # Long setup: price crosses above S3 with volume and RSI not overbought
        # Short setup: price crosses below R3 with volume and RSI not oversold
        # Breakout continuation: close beyond R4/S4
        
        long_signal = False
        short_signal = False
        
        if i > 0:  # Need previous close for crossover
            prev_close = close[i-1]
            curr_close = close[i]
            
            # Long conditions
            if (prev_close <= s3_aligned[i-1] and curr_close > s3_aligned[i] and
                volume_ok and rsi_long_ok):
                long_signal = True
            
            # Short conditions  
            if (prev_close >= r3_aligned[i-1] and curr_close < r3_aligned[i] and
                volume_ok and rsi_short_ok):
                short_signal = True
                
            # Breakout continuation (strong momentum)
            if curr_close > r4_aligned[i] and volume_ok and rsi_val > 50:
                long_signal = True
            if curr_close < s4_aligned[i] and volume_ok and rsi_val < 50:
                short_signal = True
        
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
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i))
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals