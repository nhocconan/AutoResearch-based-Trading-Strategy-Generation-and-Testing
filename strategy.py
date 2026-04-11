#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_keltner_breakout_v1"
timeframe = "12h"
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
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return signals
    
    # Calculate 20-period ATR for Keltner channels
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = abs(df_1d['high'] - df_1d['close'].shift())
    tr3 = abs(df_1d['low'] - df_1d['close'].shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_20 = tr.rolling(window=20, min_periods=20).mean().values
    
    # Calculate 20-period EMA for middle line
    ema_20 = pd.Series(df_1d['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Upper and lower Keltner channels
    keltner_upper = ema_20 + 2.0 * atr_20
    keltner_lower = ema_20 - 2.0 * atr_20
    
    # Shift by 1 to use only completed daily bars
    keltner_upper = np.roll(keltner_upper, 1)
    keltner_lower = np.roll(keltner_lower, 1)
    ema_20 = np.roll(ema_20, 1)
    keltner_upper[0] = np.nan
    keltner_lower[0] = np.nan
    ema_20[0] = np.nan
    
    # Align 1d indicators to 12h timeframe
    keltner_upper_aligned = align_htf_to_ltf(prices, df_1d, keltner_upper)
    keltner_lower_aligned = align_htf_to_ltf(prices, df_1d, keltner_lower)
    ema_20_aligned = align_htf_to_ltf(prices, df_1d, ema_20)
    
    # Calculate 12-period ATR for stoploss
    tr_12h = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    atr_12 = pd.Series(tr_12h).rolling(window=12, min_periods=12).mean().values
    
    # Volume filter: volume > 1.3x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(30, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(keltner_upper_aligned[i]) or np.isnan(keltner_lower_aligned[i]) or
            np.isnan(ema_20_aligned[i]) or np.isnan(atr_12[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        volume_current = volume[i]
        vol_ma = vol_ma_20[i]
        atr = atr_12[i]
        
        # Volume confirmation
        volume_confirmed = volume_current > 1.3 * vol_ma
        
        # Long conditions: close above upper Keltner channel with volume
        long_signal = volume_confirmed and (price_close > keltner_upper_aligned[i])
        
        # Short conditions: close below lower Keltner channel with volume
        short_signal = volume_confirmed and (price_close < keltner_lower_aligned[i])
        
        # Dynamic stoploss: 2x ATR from entry
        if position == 1 and price_close < keltner_upper_aligned[i] - 2.0 * atr:
            position = 0
            signals[i] = 0.0
        elif position == -1 and price_close > keltner_lower_aligned[i] + 2.0 * atr:
            position = 0
            signals[i] = 0.0
        else:
            # Trading logic
            if long_signal and position != 1:
                position = 1
                signals[i] = 0.25
            elif short_signal and position != -1:
                position = -1
                signals[i] = -0.25
            elif position == 1 and price_close < keltner_upper_aligned[i]:
                # Exit long if price falls back below middle line
                position = 0
                signals[i] = 0.0
            elif position == -1 and price_close > keltner_lower_aligned[i]:
                # Exit short if price rises back above middle line
                position = 0
                signals[i] = 0.0
            else:
                # Maintain current position
                signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

# Hypothesis: Keltner channel breakout on 12h with volume confirmation and ATR-based stoploss.
# Uses daily Keltner channels (EMA20 ± 2*ATR20) to identify volatility-based support/resistance.
# Enters long when price breaks above upper channel with volume confirmation (>1.3x average volume).
# Enters short when price breaks below lower channel with volume confirmation.
# Exits when price returns to middle line (EMA20) or hits 2x ATR trailing stop.
# Works in both bull and bear markets by capturing volatility expansions.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag on 12h timeframe.