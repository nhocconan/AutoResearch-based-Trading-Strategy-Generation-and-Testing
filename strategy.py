#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_donchian_volume_breakout_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return signals
    
    # Calculate 1d Donchian Channel (20-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Donchian Upper (20-period high)
    donch_high_1d = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    # Donchian Lower (20-period low)
    donch_low_1d = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Shift by 1 to use only completed 1d bars
    donch_high_1d = np.roll(donch_high_1d, 1)
    donch_low_1d = np.roll(donch_low_1d, 1)
    donch_high_1d[0] = np.nan
    donch_low_1d[0] = np.nan
    
    # Align 1d Donchian bands to 4h timeframe
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high_1d)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low_1d)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for stoploss (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    for i in range(200, n):
        # Skip if any required data is invalid
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or
            np.isnan(vol_ma_20[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        volume_current = volume[i]
        vol_ma = vol_ma_20[i]
        atr_val = atr[i]
        
        # Volume confirmation
        volume_confirmed = volume_current > 1.5 * vol_ma
        
        # Breakout conditions
        long_breakout = price_high > donch_high_aligned[i]
        short_breakout = price_low < donch_low_aligned[i]
        
        # Entry conditions
        long_signal = long_breakout and volume_confirmed
        short_signal = short_breakout and volume_confirmed
        
        # Exit conditions: opposite breakout or ATR-based stop
        exit_long = position == 1 and (price_low < donch_low_aligned[i] or 
                                       price_close < entry_price - 2.0 * atr_val)
        exit_short = position == -1 and (price_high > donch_high_aligned[i] or 
                                         price_close > entry_price + 2.0 * atr_val)
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            entry_price = price_close
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            entry_price = price_close
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

# Hypothesis: Donchian breakout strategy using 1d Donchian Channel on 4h timeframe.
# Enters long when price breaks above 1d Donchian High with volume confirmation (>1.5x avg volume).
# Enters short when price breaks below 1d Donchian Low with volume confirmation.
# Exits on opposite breakout or 2x ATR stoploss.
# Uses volume filter to reduce false breakouts and ATR for dynamic risk management.
# Designed for low trade frequency (target: 20-50 trades/year) to minimize fee drag.
# Works in both bull and bear markets by trading breakouts in either direction.