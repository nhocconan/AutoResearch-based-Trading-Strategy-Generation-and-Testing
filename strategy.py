#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout with 12h volume confirmation
# - Uses 12h Camarilla pivot levels (R3/S3 for fade, R4/S4 for breakout) with 12h volume spike
# - Long: Price breaks above R4 with volume > 2.0x 20-period average
# - Short: Price breaks below S4 with volume > 2.0x 20-period average
# - Exit: Price reverts to R3/S3 levels or opposite Camarilla breakout
# - Discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) to stay within fee drag limits
# - Camarilla pivots provide mathematical support/resistance levels that work in ranging and trending markets
# - Volume confirmation filters out false breakouts
# - 6h timeframe balances responsiveness with manageable trade frequency

name = "6h_12h_camarilla_breakout_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 12h data ONCE before loop for Camarilla pivots and volume confirmation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return signals
    
    # Pre-compute 12h Camarilla pivot levels
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate pivot point (PP)
    pp_12h = (high_12h + low_12h + close_12h) / 3.0
    
    # Calculate Camarilla levels
    r4_12h = pp_12h + ((high_12h - low_12h) * 1.1 / 2.0)
    r3_12h = pp_12h + ((high_12h - low_12h) * 1.1 / 4.0)
    s3_12h = pp_12h - ((high_12h - low_12h) * 1.1 / 4.0)
    s4_12h = pp_12h - ((high_12h - low_12h) * 1.1 / 2.0)
    
    # Pre-compute 12h volume confirmation (20-period average)
    volume_12h = df_12h['volume'].values
    volume_sma_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_aligned = align_htf_to_ltf(prices, df_12h, volume_sma_20_12h)
    
    # Align Camarilla levels to 6h timeframe
    r4_12h_aligned = align_htf_to_ltf(prices, df_12h, r4_12h)
    r3_12h_aligned = align_htf_to_ltf(prices, df_12h, r3_12h)
    s3_12h_aligned = align_htf_to_ltf(prices, df_12h, s3_12h)
    s4_12h_aligned = align_htf_to_ltf(prices, df_12h, s4_12h)
    
    for i in range(50, n):  # Start after 50-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(r4_12h_aligned[i]) or np.isnan(r3_12h_aligned[i]) or 
            np.isnan(s3_12h_aligned[i]) or np.isnan(s4_12h_aligned[i]) or
            np.isnan(volume_sma_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        volume_current = volume[i]
        
        # Camarilla levels
        r4 = r4_12h_aligned[i]
        r3 = r3_12h_aligned[i]
        s3 = s3_12h_aligned[i]
        s4 = s4_12h_aligned[i]
        
        # Volume confirmation: current volume > 2.0x 20-period average
        vol_confirm = volume_current > 2.0 * volume_sma_20_aligned[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long breakout: price closes above R4 with volume confirmation
        if close_price > r4 and vol_confirm:
            enter_long = True
        
        # Short breakout: price closes below S4 with volume confirmation
        if close_price < s4 and vol_confirm:
            enter_short = True
        
        # Exit conditions
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price reverts to R3 or breaks below S4 (opposite breakout)
            exit_long = (close_price <= r3) or (close_price < s4)
        elif position == -1:
            # Exit short if price reverts to S3 or breaks above R4 (opposite breakout)
            exit_short = (close_price >= s3) or (close_price > r4)
        
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