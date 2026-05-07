# The strategy leverages institutional price levels (Camarilla pivots) on 1d timeframe to identify key support/resistance levels. 
# A breakout above R3 or below S3 with volume confirmation triggers a position, filtered by the 12h EMA50 trend direction to avoid counter-trend trades.
# Exits occur when price crosses back below/above the 12h EMA50, ensuring trend-following behavior.
# The 12h timeframe reduces trade frequency to minimize fee drag, while the 1d pivot levels provide structure that works in both bull and bear markets.
# Position sizing is kept at 0.25 to balance return potential with drawdown control during extended bear markets like 2022.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_Camarilla_R3S3_1dEMA50_Trend"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla levels from previous day's range (using 1d data)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    prev_day_high = df_1d['high'].values
    prev_day_low = df_1d['low'].values
    prev_day_close = df_1d['close'].values
    
    # Camarilla R3 and S3 levels
    camarilla_r3 = prev_day_close + 1.1 * (prev_day_high - prev_day_low) / 2
    camarilla_s3 = prev_day_close - 1.1 * (prev_day_high - prev_day_low) / 2
    
    # Align Camarilla levels to 12h timeframe (wait for 1d bar to close)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # EMA50 on 12h close for trend filter and exit condition
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume confirmation: current volume > 2.0 * 20-period EMA of volume
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Sufficient warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema_50[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Close > Camarilla R3 AND EMA50 rising AND volume spike
            long_condition = (close[i] > camarilla_r3_aligned[i]) and (ema_50[i] > ema_50[i-1]) and volume_spike[i]
            # Short: Close < Camarilla S3 AND EMA50 falling AND volume spike
            short_condition = (close[i] < camarilla_s3_aligned[i]) and (ema_50[i] < ema_50[i-1]) and volume_spike[i]
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Close < EMA50
            if close[i] < ema_50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Close > EMA50
            if close[i] > ema_50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals