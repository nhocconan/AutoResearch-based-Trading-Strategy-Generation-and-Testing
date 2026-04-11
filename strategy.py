#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_keltner_squeeze_breakout_v1"
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
    
    # Calculate 1d Keltner Channel
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # EMA of typical price (20-period)
    tp_1d = (high_1d + low_1d + close_1d) / 3.0
    ema_tp = pd.Series(tp_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # ATR (10-period)
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_1d[0] = tr1[0]
    atr_1d = pd.Series(tr_1d).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Keltner Bands
    upper_keltner = ema_tp + (2.0 * atr_1d)
    lower_keltner = ema_tp - (2.0 * atr_1d)
    
    # Shift by 1 to use only completed 1d bars
    upper_keltner = np.roll(upper_keltner, 1)
    lower_keltner = np.roll(lower_keltner, 1)
    upper_keltner[0] = np.nan
    lower_keltner[0] = np.nan
    
    # Align 1d Keltner bands to 4h timeframe
    upper_keltner_aligned = align_htf_to_ltf(prices, df_1d, upper_keltner)
    lower_keltner_aligned = align_htf_to_ltf(prices, df_1d, lower_keltner)
    
    # Bollinger Bands on 4h for squeeze detection (20, 2)
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + (2.0 * std_20)
    lower_bb = sma_20 - (2.0 * std_20)
    
    # Squeeze condition: BB width < Keltner width
    bb_width = upper_bb - lower_bb
    kc_width = upper_keltner_aligned - lower_keltner_aligned
    squeeze = bb_width < kc_width
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(200, n):
        # Skip if any required data is invalid
        if (np.isnan(upper_keltner_aligned[i]) or np.isnan(lower_keltner_aligned[i]) or
            np.isnan(sma_20[i]) or np.isnan(std_20[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        volume_current = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume confirmation
        volume_confirmed = volume_current > 1.5 * vol_ma
        
        # Squeeze breakout conditions
        squeeze_active = squeeze[i]
        
        # Long: price breaks above upper Keltner during squeeze release with volume
        long_signal = squeeze_active and price_high > upper_keltner_aligned[i] and volume_confirmed
        
        # Short: price breaks below lower Keltner during squeeze release with volume
        short_signal = squeeze_active and price_low < lower_keltner_aligned[i] and volume_confirmed
        
        # Exit when price returns to the middle of Keltner channel (EMA of TP)
        middle_keltner_aligned = align_htf_to_ltf(prices, df_1d, ema_tp)
        if np.isnan(middle_keltner_aligned[i]):
            middle_value = price_close
        else:
            middle_value = middle_keltner_aligned[i]
        
        exit_long = position == 1 and price_close < middle_value
        exit_short = position == -1 and price_close > middle_value
        
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

# Hypothesis: Keltner squeeze breakout strategy on 4h timeframe.
# Uses 1d Keltner Channel to identify volatility contractions (squeeze) and expansions.
# Enters long when price breaks above upper Keltner Band during squeeze release with volume confirmation (>1.5x avg volume).
# Enters short when price breaks below lower Keltner Band during squeeze release with volume confirmation.
# Exits when price returns to the middle of the Keltner Channel (EMA of typical price).
# The squeeze condition (BB width < KC width) identifies low volatility periods primed for breakouts.
# Works in both bull and bear markets by trading breakouts in either direction.
# Designed for low trade frequency (target: 20-50 trades/year) to minimize fee drag.