#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_camarilla_breakout_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
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
    
    # Calculate Camarilla pivot levels for previous day
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Camarilla levels use previous day's range
    # R4 = Close + 1.5 * (High - Low)
    # R3 = Close + 1.1 * (High - Low)
    # R2 = Close + 0.55 * (High - Low)
    # R1 = Close + 0.275 * (High - Low)
    # S1 = Close - 0.275 * (High - Low)
    # S2 = Close - 0.55 * (High - Low)
    # S3 = Close - 1.1 * (High - Low)
    # S4 = Close - 1.5 * (High - Low)
    
    # Calculate daily range
    daily_range = high_1d - low_1d
    
    # Calculate levels for previous day (shift by 1 to avoid look-ahead)
    prev_close = np.roll(close_1d, 1)
    prev_range = np.roll(daily_range, 1)
    
    # Handle first element
    prev_close[0] = np.nan
    prev_range[0] = np.nan
    
    # Calculate Camarilla levels
    R4 = prev_close + 1.5 * prev_range
    R3 = prev_close + 1.1 * prev_range
    R2 = prev_close + 0.55 * prev_range
    R1 = prev_close + 0.275 * prev_range
    S1 = prev_close - 0.275 * prev_range
    S2 = prev_close - 0.55 * prev_range
    S3 = prev_close - 1.1 * prev_range
    S4 = prev_close - 1.5 * prev_range
    
    # Align levels to 4h timeframe
    R4_aligned = align_htf_to_ltf(prices, df_1d, R4)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    R2_aligned = align_htf_to_ltf(prices, df_1d, R2)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    S2_aligned = align_htf_to_ltf(prices, df_1d, S2)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    S4_aligned = align_htf_to_ltf(prices, df_1d, S4)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Trend filter: 4h EMA50
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    for i in range(60, n):  # Start after EMA50 warmup
        # Skip if any required data is invalid
        if (np.isnan(R4_aligned[i]) or np.isnan(R3_aligned[i]) or np.isnan(R2_aligned[i]) or
            np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or np.isnan(S2_aligned[i]) or
            np.isnan(S3_aligned[i]) or np.isnan(S4_aligned[i]) or np.isnan(ema_50[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        volume_current = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume confirmation
        volume_confirmed = volume_current > 1.5 * vol_ma
        
        # Long breakout: price breaks above R3 with volume
        long_breakout = volume_confirmed and (price_high > R3_aligned[i])
        
        # Short breakout: price breaks below S3 with volume
        short_breakout = volume_confirmed and (price_low < S3_aligned[i])
        
        # Trend filter: only trade in direction of EMA50
        uptrend = price_close > ema_50[i]
        downtrend = price_close < ema_50[i]
        
        # Exit conditions: price returns to middle levels
        exit_long = position == 1 and price_low < R1_aligned[i]
        exit_short = position == -1 and price_high > S1_aligned[i]
        
        # Trading logic
        if long_breakout and uptrend and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_breakout and downtrend and position != -1:
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

# Hypothesis: Camarilla pivot breakout with volume confirmation and trend filter on 4h.
# Uses daily Camarilla levels (R3/S3 for breakout, R1/S1 for exit) from previous day.
# Enters long when price breaks above R3 with volume confirmation (>1.5x average) 
# and price above 4h EMA50 (uptrend). Enters short when price breaks below S3 with
# volume confirmation and price below 4h EMA50 (downtrend). Exits when price returns
# to R1/S1 levels. Works in both bull and bear markets by filtering with EMA50 trend.
# Target: 20-50 trades/year to minimize fee drag on 4h timeframe. Camarilla levels
# provide natural support/resistance based on previous day's action. Volume confirmation
# ensures institutional participation. EMA50 filter prevents counter-trend trades.