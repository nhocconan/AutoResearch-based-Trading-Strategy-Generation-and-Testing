# The strategy uses 6h timeframe with 12h trend filter (EMA34), 12h Camarilla pivot levels, and volume confirmation
# Entry logic: Price breaks above/below S3/R3 with trend alignment and volume > 1.5x average
# Exit: Opposite Camarilla level break (S1/R1) or trend reversal
# Designed for moderate trade frequency (12-37/year) with clear risk control
# Works in bull/bear via trend filter and mean-reversion at extreme pivot levels

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for higher timeframe context
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h_series = pd.Series(close_12h)
    
    # Calculate 12h EMA 34 for trend direction
    ema_34_12h = close_12h_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Calculate 12h Camarilla pivot levels
    # Camarilla formulas: 
    # R4 = close + ((high - low) * 1.1/2)
    # R3 = close + ((high - low) * 1.1/4)
    # R2 = close + ((high - low) * 1.1/6)
    # R1 = close + ((high - low) * 1.1/12)
    # S1 = close - ((high - low) * 1.1/12)
    # S2 = close - ((high - low) * 1.1/6)
    # S3 = close - ((high - low) * 1.1/4)
    # S4 = close - ((high - low) * 1.1/2)
    diff_12h = (high_12h - low_12h)
    r3_12h = close_12h + (diff_12h * 1.1 / 4)
    s3_12h = close_12h - (diff_12h * 1.1 / 4)
    r1_12h = close_12h + (diff_12h * 1.1 / 12)
    s1_12h = close_12h - (diff_12h * 1.1 / 12)
    
    # Align Camarilla levels to 6h timeframe
    r3_12h_aligned = align_htf_to_ltf(prices, df_12h, r3_12h)
    s3_12h_aligned = align_htf_to_ltf(prices, df_12h, s3_12h)
    r1_12h_aligned = align_htf_to_ltf(prices, df_12h, r1_12h)
    s1_12h_aligned = align_htf_to_ltf(prices, df_12h, s1_12h)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_12h_aligned[i]) or 
            np.isnan(r3_12h_aligned[i]) or np.isnan(s3_12h_aligned[i]) or
            np.isnan(r1_12h_aligned[i]) or np.isnan(s1_12h_aligned[i]) or
            np.isnan(vol_ma[i]) or np.isnan(volume_filter[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 12h EMA34
        price_above_ema = close[i] > ema_34_12h_aligned[i]
        price_below_ema = close[i] < ema_34_12h_aligned[i]
        
        # Volume condition
        vol_ok = volume_filter[i]
        
        # Long conditions: price breaks above S3 with uptrend and volume
        long_signal = (close[i] > s3_12h_aligned[i-1] and price_above_ema and vol_ok)
        # Short conditions: price breaks below R3 with downtrend and volume
        short_signal = (close[i] < r3_12h_aligned[i-1] and price_below_ema and vol_ok)
        
        if long_signal:
            signals[i] = 0.25
            position = 1
        elif short_signal:
            signals[i] = -0.25
            position = -1
        # Exit conditions: 
        # 1. Opposite Camarilla level break (S1 for longs, R1 for shorts)
        # 2. Trend reversal
        elif position == 1 and (close[i] < s1_12h_aligned[i-1] or not price_above_ema):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (close[i] > r1_12h_aligned[i-1] or not price_below_ema):
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_Camarilla_S3R3_Breakout_12hEMA34_VolumeFilter"
timeframe = "6h"
leverage = 1.0