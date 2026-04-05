#!/usr/bin/env python3
"""
Experiment #8467: 6h Donchian breakout + daily Camarilla pivot + volume confirmation
Hypothesis: Daily Camarilla pivot levels (R3/S3 fade, R4/S4 breakout) provide institutional reference points. 
Combined with 6h Donchian breakouts and volume confirmation, this creates a regime-aware strategy 
that fades extreme reversions at S3/R3 and continues breaks beyond S4/R4. Works in both bull/bear 
markets by adapting to price action relative to daily pivot structure.
"""

from mtf_data import get_athf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_8467_6h_donchian20_1d_camarilla_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
CAMARILLA_LOOKBACK = 1  # Use previous day's OHLC
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_camarilla(high, low, close):
    """
    Calculate Camarilla pivot levels for given OHLC
    R4 = Close + (High-Low) * 1.1/2
    R3 = Close + (High-Low) * 1.1/4
    S3 = Close - (High-Low) * 1.1/4
    S4 = Close - (High-Low) * 1.1/2
    """
    range_hl = high - low
    r4 = close + range_hl * 1.1 / 2
    r3 = close + range_hl * 1.1 / 4
    s3 = close - range_hl * 1.1 / 4
    s4 = close - range_hl * 1.1 / 2
    return r3, r4, s3, s4

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily Camarilla levels from previous day's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Shift by 1 to use only completed daily bars (no look-ahead)
    r3, r4, s3, s4 = calculate_camarilla(high_1d, low_1d, close_1d)
    r3 = np.roll(r3, 1)
    r4 = np.roll(r4, 1)
    s3 = np.roll(s3, 1)
    s4 = np.roll(s4, 1)
    # Set first value to NaN (no previous day)
    r3[0] = np.nan
    r4[0] = np.nan
    s3[0] = np.nan
    s4[0] = np.nan
    
    # Align Camarilla levels to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Calculate LTF indicators (6h)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels
    donchian_high = pd.Series(high).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max().values
    donchian_low = pd.Series(low).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min().values
    
    # Volume moving average
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR for risk management
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(r3_aligned[i]) or np.isnan(r4_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]):
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
        
        # Price action relative to Camarilla levels
        price = close[i]
        
        # Fade at S3/R3 (mean reversion)
        fade_long = price <= s3_aligned[i]  # At or below S3 - potential bounce
        fade_short = price >= r3_aligned[i]  # At or above R3 - potential rejection
        
        # Breakout continuation beyond S4/R4 (trend follow)
        breakout_long = price >= r4_aligned[i]  # Above R4 - bullish breakout
        breakout_short = price <= s4_aligned[i]  # Below S4 - bearish breakdown
        
        # Volume confirmation
        volume_confirmed = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Entry conditions: fade at S3/R3 OR breakout beyond S4/R4 with volume
        long_entry = (fade_long or breakout_long) and volume_confirmed
        short_entry = (fade_short or breakout_short) and volume_confirmed
        
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