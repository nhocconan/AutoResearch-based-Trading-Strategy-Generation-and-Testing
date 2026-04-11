#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_camarilla_breakout_volume_v4"
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
    if len(df_1d) < 50:
        return signals
    
    # Calculate Camarilla pivot levels from daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla formula: range = high - low
    # Resistance levels: R1 = close + (range * 1.1/12), R2 = close + (range * 1.1/6), R3 = close + (range * 1.1/4), R4 = close + (range * 1.1/2)
    # Support levels: S1 = close - (range * 1.1/12), S2 = close - (range * 1.1/6), S3 = close - (range * 1.1/4), S4 = close - (range * 1.1/2)
    daily_range = high_1d - low_1d
    
    # Key levels for breakout: R4 (resistance) and S4 (support)
    r4 = close_1d + (daily_range * 1.1 / 2)
    s4 = close_1d - (daily_range * 1.1 / 2)
    
    # Exit levels: R3 and S3
    r3 = close_1d + (daily_range * 1.1 / 4)
    s3 = close_1d - (daily_range * 1.1 / 4)
    
    # Volume confirmation: 12h volume > 2.5x 20-period average (very strict to reduce trades)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR-based volatility filter: ATR(14) > 0.5 * 20-period ATR average (avoid low volatility periods)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]  # first element
    tr3[0] = tr1[0]  # first element
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma = pd.Series(atr).rolling(window=20, min_periods=20).mean().values
    vol_filter = atr > 0.5 * atr_ma
    
    # Align daily levels to 12h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(vol_ma_20[i]) or np.isnan(atr[i]) or np.isnan(atr_ma[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        volume_current = volume[i]
        
        # Volume confirmation - very strict
        vol_confirm = volume_current > 2.5 * vol_ma_20[i]
        
        # Volatility filter - avoid low volatility periods
        vol_filter_pass = vol_filter[i]
        
        # Breakout conditions using Camarilla levels
        breakout_up = price_close > r4_aligned[i]  # Break above R4
        breakout_down = price_close < s4_aligned[i]  # Break below S4
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Break above R4 with volume confirmation and volatility filter
        if breakout_up and vol_confirm and vol_filter_pass:
            enter_long = True
        
        # Short: Break below S4 with volume confirmation and volatility filter
        if breakout_down and vol_confirm and vol_filter_pass:
            enter_short = True
        
        # Exit conditions: return to opposite S3/R3 levels
        exit_long = price_close < s3_aligned[i]  # Return to S3 level
        exit_short = price_close > r3_aligned[i]  # Return to R3 level
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.30
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.30
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.30 if position == 1 else (-0.30 if position == -1 else 0.0)
    
    return signals

# Hypothesis: 12h Camarilla breakout strategy using daily pivot levels with volume confirmation and volatility filter.
# Enters long when price breaks above R4 with volume > 2.5x 20-period average and ATR > 0.5x 20-period ATR average.
# Enters short when price breaks below S4 with volume > 2.5x 20-period average and ATR > 0.5x 20-period ATR average.
# Exits when price returns to S3/R3 levels respectively.
# Uses very strict volume threshold (2.5x) and volatility filter to achieve 10-20 trades per year.
# Position size set to 0.30 to balance risk and reward.
# Target: 10-20 trades per year (40-80 total over 4 years) to minimize fee drag.
# Works in both bull and bear markets by capturing significant breakouts in either direction.
# 12h timeframe reduces noise and 1d Camarilla levels provide institutional reference points.