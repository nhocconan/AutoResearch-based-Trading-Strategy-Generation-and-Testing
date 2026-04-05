#!/usr/bin/env python3
"""
exp_6899_6h_donchian20_12h_pivot_vol_v1
Hypothesis: 6h Donchian(20) breakout with 12h Camarilla pivot levels and volume confirmation.
Go long when price breaks above R4 in uptrend (close > 12h EMA50), short when breaks below S4 in downtrend (close < 12h EMA50).
Fade at R3/S3 in ranging markets (ADX < 25). Volume confirms breakout legitimacy.
Designed for 6h timeframe to capture swings with ~12-37 trades/year (50-150 total over 4 years).
Works in both bull and bear markets by using 12h EMA for trend and ADX for regime detection.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_6899_6h_donchian20_12h_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
VOL_MA_PERIOD = 20
VOL_BASE_THRESHOLD = 2.0
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5
MAX_HOLD_BARS = 20  # ~5 days (6h bars)
EMA_PERIOD = 50
ADX_PERIOD = 14
ADX_THRESHOLD = 25
PIVOT_LOOKBACK = 10  # periods for pivot calculation

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 12h for EMA, ADX, and pivot
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h EMA50
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=EMA_PERIOD, adjust=False, min_periods=EMA_PERIOD).mean().values
    
    # Calculate 12h ADX
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = pd.Series(high_12h - low_12h)
    tr2 = pd.Series(np.abs(high_12h - np.roll(close_12h, 1)))
    tr3 = pd.Series(np.abs(low_12h - np.roll(close_12h, 1)))
    tr_12h = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    dm_plus = pd.Series(np.where((high_12h - np.roll(high_12h, 1)) > (np.roll(low_12h, 1) - low_12h),
                                 np.maximum(high_12h - np.roll(high_12h, 1), 0), 0))
    dm_minus = pd.Series(np.where((np.roll(low_12h, 1) - low_12h) > (high_12h - np.roll(high_12h, 1)),
                                  np.maximum(np.roll(low_12h, 1) - low_12h, 0), 0))
    
    # Smoothed values
    tr_14 = tr_12h.ewm(span=ADX_PERIOD, adjust=False, min_periods=ADX_PERIOD).mean()
    dm_plus_14 = dm_plus.ewm(span=ADX_PERIOD, adjust=False, min_periods=ADX_PERIOD).mean()
    dm_minus_14 = dm_minus.ewm(span=ADX_PERIOD, adjust=False, min_periods=ADX_PERIOD).mean()
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = dx.ewm(span=ADX_PERIOD, adjust=False, min_periods=ADX_PERIOD).mean()
    
    adx_values = adx.values
    di_plus_values = di_plus.values
    di_minus_values = di_minus.values
    
    # Calculate 12h Camarilla pivot levels (based on previous day's OHLC)
    # Camarilla: R4 = C + ((H-L) * 1.1/2), R3 = C + ((H-L) * 1.1/4), etc.
    # We'll use the previous 12h bar's OHLC to calculate pivot for current bar
    open_12h = df_12h['open'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Previous bar's OHLC for pivot calculation
    prev_open = np.roll(open_12h, 1)
    prev_high = np.roll(high_12h, 1)
    prev_low = np.roll(low_12h, 1)
    prev_close = np.roll(close_12h, 1)
    
    # Camarilla levels
    camarilla_r4 = prev_close + ((prev_high - prev_low) * 1.1 / 2)
    camarilla_r3 = prev_close + ((prev_high - prev_low) * 1.1 / 4)
    camarilla_s3 = prev_close - ((prev_high - prev_low) * 1.1 / 4)
    camarilla_s4 = prev_close - ((prev_high - prev_low) * 1.1 / 2)
    
    # Align HTF indicators to LTF (6h)
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx_values)
    di_plus_aligned = align_htf_to_ltf(prices, df_12h, di_plus_values)
    di_minus_aligned = align_htf_to_ltf(prices, df_12h, di_minus_values)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r4)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s3)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s4)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels
    highest_high = pd.Series(high).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max().values
    lowest_low = pd.Series(low).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min().values
    
    # Volume MA for confirmation
    vol_ma = pd.Series(volume).rolling(window=VOL_MA_PERIOD, min_periods=VOL_MA_PERIOD).mean().values
    
    # ATR for stoploss
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, VOL_MA_PERIOD, ATR_PERIOD, EMA_PERIOD, ADX_PERIOD, PIVOT_LOOKBACK) + 1
    
    for i in range(start, n):
        bars_since_entry += 1
        
        # Skip if HTF data not available
        if (np.isnan(ema_12h_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_s4_aligned[i])):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= entry_price - ATR_STOP_MULTIPLIER * atr[i]:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
                continue
        elif position == -1:  # short position
            if close[i] >= entry_price + ATR_STOP_MULTIPLIER * atr[i]:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
                continue
                
        # Time-based exit
        if position != 0 and bars_since_entry >= MAX_HOLD_BARS:
            signals[i] = 0.0
            position = 0
            bars_since_entry = 0
            continue
            
        # Volume confirmation
        vol_confirmed = volume[i] > vol_ma[i] * VOL_BASE_THRESHOLD if not np.isnan(vol_ma[i]) else False
        
        # Determine trend direction from 12h EMA50 and ADX
        uptrend = (close[i] > ema_12h_aligned[i]) and (adx_aligned[i] > ADX_THRESHOLD) and (di_plus_aligned[i] > di_minus_aligned[i])
        downtrend = (close[i] < ema_12h_aligned[i]) and (adx_aligned[i] > ADX_THRESHOLD) and (di_minus_aligned[i] > di_plus_aligned[i])
        ranging = adx_aligned[i] < ADX_THRESHOLD
        
        # Breakout signals (only in trending markets)
        long_breakout = uptrend and (close[i] > camarilla_r4_aligned[i]) and vol_confirmed
        short_breakout = downtrend and (close[i] < camarilla_s4_aligned[i]) and vol_confirmed
        
        # Fade signals (only in ranging markets)
        long_fade = ranging and (close[i] < camarilla_s3_aligned[i]) and vol_confirmed
        short_fade = ranging and (close[i] > camarilla_r3_aligned[i]) and vol_confirmed
        
        # Enter new positions only if flat
        if position == 0:
            if long_breakout:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                bars_since_entry = 0
            elif short_breakout:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                bars_since_entry = 0
            elif long_fade:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                bars_since_entry = 0
            elif short_fade:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                bars_since_entry = 0
            else:
                signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = position * SIGNAL_SIZE
    
    return signals