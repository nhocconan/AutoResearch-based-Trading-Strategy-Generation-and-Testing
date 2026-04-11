#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_keltner_breakout_volume_v1"
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
    
    # Calculate 1d Keltner Channel components
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 20-period EMA of close
    ema_20 = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Average True Range (ATR) over 10 periods
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_10 = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Keltner Channel bounds
    upper_keltner = ema_20 + 2.0 * atr_10
    lower_keltner = ema_20 - 2.0 * atr_10
    
    # Shift by 1 to use only completed 1d bars
    upper_keltner = np.roll(upper_keltner, 1)
    lower_keltner = np.roll(lower_keltner, 1)
    ema_20 = np.roll(ema_20, 1)
    upper_keltner[0] = np.nan
    lower_keltner[0] = np.nan
    ema_20[0] = np.nan
    
    # Align 1d indicators to 12h timeframe
    upper_keltner_aligned = align_htf_to_ltf(prices, df_1d, upper_keltner)
    lower_keltner_aligned = align_htf_to_ltf(prices, df_1d, lower_keltner)
    ema_20_aligned = align_htf_to_ltf(prices, df_1d, ema_20)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(60, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(upper_keltner_aligned[i]) or np.isnan(lower_keltner_aligned[i]) or
            np.isnan(ema_20_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        volume_current = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume confirmation
        volume_confirmed = volume_current > 1.5 * vol_ma
        
        # Long conditions: Price breaks above upper Keltner with volume
        long_signal = volume_confirmed and (price_high > upper_keltner_aligned[i])
        
        # Short conditions: Price breaks below lower Keltner with volume
        short_signal = volume_confirmed and (price_low < lower_keltner_aligned[i])
        
        # Exit when price crosses back to EMA
        exit_long = position == 1 and (price_close < ema_20_aligned[i])
        exit_short = position == -1 and (price_close > ema_20_aligned[i])
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
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

# Hypothesis: Keltner breakout on 12h with daily Keltner Channel and volume confirmation.
# Uses daily Keltner Channel (EMA20 ± 2*ATR10) to identify volatility breakouts.
# Enters long when 12h high breaks above daily upper Keltner with volume confirmation
# (>1.5x average volume). Enters short when 12h low breaks below daily lower Keltner
# with volume confirmation. Exits when price crosses back to the daily EMA20.
# Works in both bull and bear markets by capturing volatility expansion phases.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag on 12h timeframe.
# Volume confirmation ensures participation from market actors, reducing false breakouts.
# Keltner channels adapt to volatility, making them effective in various market conditions.