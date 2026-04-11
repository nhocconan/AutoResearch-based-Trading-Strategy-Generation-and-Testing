#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h_daily_camarilla_breakout_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
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
    
    daily_range = high_1d - low_1d
    
    # Key levels: R4 (resistance) and S4 (support)
    r4 = close_1d + (daily_range * 1.1 / 2)
    s4 = close_1d - (daily_range * 1.1 / 2)
    
    # Exit levels: R3 and S3
    r3 = close_1d + (daily_range * 1.1 / 4)
    s3 = close_1d - (daily_range * 1.1 / 4)
    
    # Align daily levels to 1h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # 1h volume confirmation: current volume > 2.0x 50-period average
    vol_ma_50 = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(vol_ma_50[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        volume_current = volume[i]
        
        # Volume confirmation
        vol_confirm = volume_current > 2.0 * vol_ma_50[i]
        
        # Breakout conditions using daily Camarilla levels
        breakout_up = price_close > r4_aligned[i]   # Break above daily R4
        breakout_down = price_close < s4_aligned[i] # Break below daily S4
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Break above daily R4 with volume confirmation
        if breakout_up and vol_confirm:
            enter_long = True
        
        # Short: Break below daily S4 with volume confirmation
        if breakout_down and vol_confirm:
            enter_short = True
        
        # Exit conditions: return to opposite R3/S3 levels
        exit_long = price_close < s3_aligned[i]  # Return to daily S3 level
        exit_short = price_close > r3_aligned[i] # Return to daily R3 level
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.20
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.20
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.20 if position == 1 else (-0.20 if position == -1 else 0.0)
    
    return signals

# Hypothesis: 1h Camarilla breakout using daily pivot levels with volume confirmation.
# Enters long when price breaks above daily R4 with volume > 2.0x 50-period average.
# Enters short when price breaks below daily S4 with volume > 2.0x 50-period average.
# Exits when price returns to daily S3/R3 levels respectively.
# Uses daily Camarilla levels for higher timeframe context, reducing false breakouts.
# Volume filter ensures only significant breakouts are traded.
# Position size fixed at 0.20 to manage risk and reduce drawdown.
# Target: 15-30 trades per year (60-120 total over 4 years) to minimize fee drag.
# Works in both bull and bear markets by capturing significant breakouts in either direction.
# Uses 1h timeframe for precise entry timing while relying on daily structure for direction.