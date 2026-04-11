#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_camarilla_breakout_volume_v1"
timeframe = "4h"
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
    
    # Calculate daily Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's values (Camarilla uses previous day's OHLC)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    # Set first day to NaN (no previous day)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Camarilla levels for previous day
    # Resistance levels
    R4 = prev_close + (prev_high - prev_low) * 1.500
    R3 = prev_close + (prev_high - prev_low) * 1.250
    R2 = prev_close + (prev_high - prev_low) * 1.166
    R1 = prev_close + (prev_high - prev_low) * 1.083
    # Support levels
    S1 = prev_close - (prev_high - prev_low) * 1.083
    S2 = prev_close - (prev_high - prev_low) * 1.166
    S3 = prev_close - (prev_high - prev_low) * 1.250
    S4 = prev_close - (prev_high - prev_low) * 1.500
    
    # Align to 4h timeframe
    R4_aligned = align_htf_to_ltf(prices, df_1d, R4)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    R2_aligned = align_htf_to_ltf(prices, df_1d, R2)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    S2_aligned = align_htf_to_ltf(prices, df_1d, S2)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    S4_aligned = align_htf_to_ltf(prices, df_1d, S4)
    
    # Volume confirmation: 4h volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(R4_aligned[i]) or np.isnan(R3_aligned[i]) or np.isnan(R2_aligned[i]) or
            np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or np.isnan(S2_aligned[i]) or
            np.isnan(S3_aligned[i]) or np.isnan(S4_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        volume_current = volume[i]
        
        # Volume confirmation
        vol_confirm = volume_current > 1.5 * vol_ma_20[i]
        
        # Camarilla breakout conditions
        # Long: price breaks above R3 with volume
        enter_long = (price_close > R3_aligned[i]) and vol_confirm
        # Short: price breaks below S3 with volume
        enter_short = (price_close < S3_aligned[i]) and vol_confirm
        
        # Exit conditions: price returns to median (close to previous day's close)
        exit_long = price_close < prev_close[i] if not np.isnan(prev_close[i]) else False
        exit_short = price_close > prev_close[i] if not np.isnan(prev_close[i]) else False
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif enter_short and position != -1:
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

# Hypothesis: Camarilla pivot breakouts on daily timeframe with 4h execution.
# Camarilla levels identify key support/resistance levels based on previous day's range.
# Breakouts above R3 or below S3 with volume confirmation indicate strong momentum.
# Returns to previous day's close act as natural profit targets/reversal points.
# Works in both bull and bear markets as it captures breakout moves in either direction.
# Position size 0.25 limits drawdown. Target: 20-50 trades per year.