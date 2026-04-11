# 6h_1d_camarilla_pivot_breakout_v1
# Hypothesis: 6h Camarilla pivot levels from 1d with volume confirmation
# - Long when price breaks above R3 (resistance level 3) with volume > 1.5x 20-period average
# - Short when price breaks below S3 (support level 3) with volume > 1.5x 20-period average
# - Exit when price reverses to opposite pivot level (R2 for long, S2 for short) or volatility drops
# - Uses daily Camarilla levels for structure, 6h for execution timing
# - Volume filter ensures breakouts have conviction
# - Target: 12-37 trades/year (50-150 total over 4 years) to stay within 6h limits

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_camarilla_pivot_breakout_v1"
timeframe = "6h"
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
    
    # Load 1d data ONCE before loop for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return signals
    
    # Pre-calculate 1d Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla formulas
    # Range = High - Low
    # R4 = Close + Range * 1.1/2
    # R3 = Close + Range * 1.1/4
    # R2 = Close + Range * 1.1/6
    # R1 = Close + Range * 1.1/12
    # S1 = Close - Range * 1.1/12
    # S2 = Close - Range * 1.1/6
    # S3 = Close - Range * 1.1/4
    # S4 = Close - Range * 1.1/2
    range_1d = high_1d - low_1d
    r3_1d = close_1d + range_1d * 1.1 / 4
    s3_1d = close_1d - range_1d * 1.1 / 4
    r2_1d = close_1d + range_1d * 1.1 / 6
    s2_1d = close_1d - range_1d * 1.1 / 6
    
    # Align Camarilla levels to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2_1d)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2_1d)
    
    # Pre-compute 6h volume SMA (20-period)
    volume_series = pd.Series(volume)
    volume_sma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or
            np.isnan(volume_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        volume_current = volume[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume_current > 1.5 * volume_sma_20[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Price breaks above R3 with volume confirmation
        if price_high > r3_aligned[i] and vol_confirm:
            enter_long = True
        
        # Short: Price breaks below S3 with volume confirmation
        if price_low < s3_aligned[i] and vol_confirm:
            enter_short = True
        
        # Exit conditions
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price drops back to R2 (profit target) or breaks S3 (stop)
            exit_long = price_close < r2_aligned[i] or price_low < s3_aligned[i]
        elif position == -1:
            # Exit short if price rises to S2 (profit target) or breaks R3 (stop)
            exit_short = price_close > s2_aligned[i] or price_high > r3_aligned[i]
        
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