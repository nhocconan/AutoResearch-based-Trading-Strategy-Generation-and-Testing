#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_keltner_breakout_v1"
timeframe = "1d"
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
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return signals
    
    # Calculate weekly ATR (20-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First TR is just high-low
    
    atr_20 = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    # Keltner Channels: 20-period EMA ± 2 * ATR
    ema_20 = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    upper_keltner = ema_20 + 2 * atr_20
    lower_keltner = ema_20 - 2 * atr_20
    
    # Shift by 1 to use only completed weekly bars
    upper_keltner = np.roll(upper_keltner, 1)
    lower_keltner = np.roll(lower_keltner, 1)
    ema_20 = np.roll(ema_20, 1)
    upper_keltner[0] = np.nan
    lower_keltner[0] = np.nan
    ema_20[0] = np.nan
    
    # Align weekly Keltner channels to daily timeframe
    upper_keltner_aligned = align_htf_to_ltf(prices, df_1w, upper_keltner)
    lower_keltner_aligned = align_htf_to_ltf(prices, df_1w, lower_keltner)
    ema_20_aligned = align_htf_to_ltf(prices, df_1w, ema_20)
    
    # Daily volume filter: volume > 1.8x 20-day average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Daily close for breakout confirmation
    close_daily = close
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(upper_keltner_aligned[i]) or np.isnan(lower_keltner_aligned[i]) or
            np.isnan(ema_20_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price_close = close_daily[i]
        volume_current = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume confirmation
        volume_confirmed = volume_current > 1.8 * vol_ma
        
        # Long breakout: price closes above upper Keltner band
        long_signal = volume_confirmed and (price_close > upper_keltner_aligned[i])
        
        # Short breakout: price closes below lower Keltner band
        short_signal = volume_confirmed and (price_close < lower_keltner_aligned[i])
        
        # Exit when price returns to middle (EMA20)
        exit_long = position == 1 and price_close < ema_20_aligned[i]
        exit_short = position == -1 and price_close > ema_20_aligned[i]
        
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

# Hypothesis: Weekly Keltner Channel breakout with volume confirmation on daily timeframe.
# Uses weekly ATR-based channels (EMA20 ± 2*ATR(20)) to identify volatility expansion.
# Enters long when daily close breaks above upper weekly Keltner band with volume > 1.8x average.
# Enters short when daily close breaks below lower weekly Keltner band with volume confirmation.
# Exits when price returns to the weekly EMA20 (middle of Keltner channel).
# Works in both bull and bear markets by capturing volatility expansion breakouts.
# Weekly timeframe filters noise, daily timeframe provides timely execution.
# Target: 20-60 trades over 4 years (5-15/year) to minimize fee drag on 1d timeframe.