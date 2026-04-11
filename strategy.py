#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_camarilla_breakout_volume_trend_v1"
timezone = "UTC"
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
    if len(df_1d) < 20:
        return signals
    
    # Calculate daily Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point
    pivot = (high_1d + low_1d + close_1d) / 3.0
    # Range
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    r4 = close_1d + (range_1d * 1.1 / 2)
    r3 = close_1d + (range_1d * 1.1 / 4)
    r2 = close_1d + (range_1d * 1.1 / 6)
    r1 = close_1d + (range_1d * 1.1 / 12)
    s1 = close_1d - (range_1d * 1.1 / 12)
    s2 = close_1d - (range_1d * 1.1 / 6)
    s3 = close_1d - (range_1d * 1.1 / 4)
    s4 = close_1d - (range_1d * 1.1 / 2)
    
    # Shift by 1 to use only completed daily bars
    pivot = np.roll(pivot, 1)
    r1 = np.roll(r1, 1)
    r2 = np.roll(r2, 1)
    r3 = np.roll(r3, 1)
    r4 = np.roll(r4, 1)
    s1 = np.roll(s1, 1)
    s2 = np.roll(s2, 1)
    s3 = np.roll(s3, 1)
    s4 = np.roll(s4, 1)
    pivot[0] = np.nan
    r1[0] = np.nan
    r2[0] = np.nan
    r3[0] = np.nan
    r4[0] = np.nan
    s1[0] = np.nan
    s2[0] = np.nan
    s3[0] = np.nan
    s4[0] = np.nan
    
    # Align daily levels to 12h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Calculate 12h EMA25 for trend filter
    ema_25 = pd.Series(close).ewm(span=25, adjust=False, min_periods=25).mean().values
    
    # Volume filter: volume > 1.8x 30-period average
    vol_ma_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    
    # Calculate 12h ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = np.nan  # First TR is invalid
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    for i in range(30, n):  # Start after EMA25 and ATR warmup
        # Skip if any required data is invalid
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(r2_aligned[i]) or
            np.isnan(r3_aligned[i]) or np.isnan(r4_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(s2_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(ema_25[i]) or np.isnan(vol_ma_30[i]) or np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        volume_current = volume[i]
        vol_ma = vol_ma_30[i]
        atr = atr_14[i]
        
        # Volume confirmation
        volume_confirmed = volume_current > 1.8 * vol_ma
        
        # Long conditions: price breaks above R3 with volume and uptrend
        long_breakout = price_high > r3_aligned[i]
        long_trend = price_close > ema_25[i]
        long_signal = volume_confirmed and long_breakout and long_trend
        
        # Short conditions: price breaks below S3 with volume and downtrend
        short_breakout = price_low < s3_aligned[i]
        short_trend = price_close < ema_25[i]
        short_signal = volume_confirmed and short_breakout and short_trend
        
        # Stoploss: 2 * ATR from entry
        stop_long = position == 1 and price_close < ema_25[i] - 2.0 * atr
        stop_short = position == -1 and price_close > ema_25[i] + 2.0 * atr
        
        # Exit when price returns to pivot (mean reversion)
        exit_long = position == 1 and price_close < pivot_aligned[i]
        exit_short = position == -1 and price_close > pivot_aligned[i]
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and (exit_long or stop_long):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (exit_short or stop_short):
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

# Hypothesis: Camarilla pivot breakout with volume confirmation and EMA25 trend filter on 12h.
# Uses daily Camarilla levels (R3/S3 as key resistance/support) for breakout signals.
# Enters long when price breaks above R3 with volume confirmation (>1.8x average) and
# price above EMA25 (uptrend). Enters short when price breaks below S3 with volume
# confirmation and price below EMA25 (downtrend). Exits when price returns to the
# daily pivot or when stoploss (2*ATR) is hit. Works in both bull and bear markets
# by trading breakouts in the direction of the 12h EMA25 trend. Target: 50-150 total
# trades over 4 years (12-37/year) to minimize fee drag on 12h timeframe. Camarilla
# levels provide mathematically derived support/resistance that works across market
# regimes. Volume confirmation ensures institutional participation. EMA25 filter
# prevents counter-trend trades and ATR-based stoploss manages risk.