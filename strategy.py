#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_camarilla_breakout_volume_v2"
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
    
    # Calculate weekly Camarilla pivot levels
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    weekly_range = high_1w - low_1w
    r4 = close_1w + (weekly_range * 1.1 / 2)
    s4 = close_1w - (weekly_range * 1.1 / 2)
    r3 = close_1w + (weekly_range * 1.1 / 4)
    s3 = close_1w - (weekly_range * 1.1 / 4)
    
    # Volume confirmation: daily volume > 2.0x 50-period average
    vol_ma_50 = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    
    # Align weekly levels to daily timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1w, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1w, s4)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(vol_ma_50[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        volume_current = volume[i]
        
        # Volume confirmation - strict to reduce trades
        vol_confirm = volume_current > 2.0 * vol_ma_50[i]
        
        # Breakout conditions using weekly Camarilla levels
        breakout_up = price_close > r4_aligned[i]
        breakout_down = price_close < s4_aligned[i]
        
        # Entry conditions
        enter_long = breakout_up and vol_confirm
        enter_short = breakout_down and vol_confirm
        
        # Exit conditions: return to opposite S3/R3 levels
        exit_long = price_close < s3_aligned[i]
        exit_short = price_close > r3_aligned[i]
        
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

# Hypothesis: 1d Camarilla breakout strategy using weekly pivot levels with volume confirmation.
# Enters long when price breaks above weekly R4 with volume > 2.0x 50-period average.
# Enters short when price breaks below weekly S4 with volume > 2.0x 50-period average.
# Exits when price returns to weekly S3/R3 levels respectively.
# Uses strict volume threshold (2.0x) to achieve 10-25 trades per year.
# Position size set to 0.30 to balance risk and reward.
# Weekly timeframe provides institutional reference points that work in both bull and bear markets.
# Target: 10-25 trades per year (40-100 total over 4 years) to minimize fee drag.