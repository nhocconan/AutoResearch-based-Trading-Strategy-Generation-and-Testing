#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h_1d_camarilla_breakout_v2"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    # Calculate daily pivot points (Camarilla)
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close[0] = close[0]
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    
    pivot = (prev_high + prev_low + prev_close) / 3
    range_val = prev_high - prev_low
    
    # Camarilla levels
    r4 = pivot + (range_val * 1.5)
    r3 = pivot + (range_val * 1.25)
    s3 = pivot - (range_val * 1.25)
    s4 = pivot - (range_val * 1.5)
    
    # ATR for volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma = pd.Series(atr).rolling(window=50, min_periods=50).mean().values
    atr_ratio = atr / (atr_ma + 1e-10)
    low_vol = atr_ratio < 0.5  # Low volatility filter
    
    # Volume confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_surge = volume > (vol_ma * 1.5)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(r4[i]) or np.isnan(r3[i]) or np.isnan(s3[i]) or np.isnan(s4[i]) or
            np.isnan(atr[i]) or np.isnan(vol_ma[i]) or np.isnan(hours[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        vol_surge_now = vol_surge[i]
        low_vol_now = low_vol[i]
        session_now = session_filter[i]
        
        # Entry conditions - require low volatility + volume surge + session
        long_signal = (price_close > r4[i]) and vol_surge_now and low_vol_now and session_now
        short_signal = (price_close < s3[i]) and vol_surge_now and low_vol_now and session_now
        
        # Exit conditions - reverse at opposite Camarilla level
        exit_long = position == 1 and price_close < r3[i]
        exit_short = position == -1 and price_close > s4[i]
        
        # Stop loss - 2x ATR
        stop_long = position == 1 and price_low < (entry_price - 2.0 * atr[i])
        stop_short = position == -1 and price_high > (entry_price + 2.0 * atr[i])
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            entry_price = price_close
            signals[i] = 0.20
        elif short_signal and position != -1:
            position = -1
            entry_price = price_close
            signals[i] = -0.20
        elif position == 1 and (exit_long or stop_long):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (exit_short or stop_short):
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.20 if position == 1 else (-0.20 if position == -1 else 0.0)
    
    return signals

# Hypothesis: 1h Camarilla breakout strategy with volume surge and low volatility filter.
# Enters long when price breaks above Camarilla R4 level with volume surge (>1.5x avg volume) in low volatility conditions (ATR ratio < 0.5) during active session (08-20 UTC).
# Enters short when price breaks below Camarilla S3 level with same conditions.
# Uses Camarilla R3/S4 for exits and 2x ATR for stop loss.
# Designed for 1h timeframe to target 60-150 total trades over 4 years (15-37/year).
# Works in both bull and bear markets by capturing breakouts in either direction with volatility filter to avoid whipsaws.