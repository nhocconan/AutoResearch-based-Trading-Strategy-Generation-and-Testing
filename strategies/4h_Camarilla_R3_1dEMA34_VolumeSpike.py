#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3 level breakout with 1d EMA34 trend filter and volume spike confirmation.
# Camarilla pivot levels provide precise reversal points in ranging markets; EMA34 on 1d confirms trend direction.
# Volume spikes (>2x average) confirm institutional interest. Designed for low trade frequency to minimize fee drag.
name = "4h_Camarilla_R3_1dEMA34_VolumeSpike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 34-period EMA on 1d close
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla pivot levels from previous 1d bar
    # Typical price, range, and levels based on previous day
    typical_price = (high + low + close) / 3
    range_val = high - low
    
    # R3 level: Close + 1.1 * (High - Low) / 2
    r3 = close + 1.1 * range_val / 2
    # S3 level: Close - 1.1 * (High - Low) / 2
    s3 = close - 1.1 * range_val / 2
    
    # Shift to align with current bar (use previous day's levels)
    r3 = np.roll(r3, 1)
    s3 = np.roll(s3, 1)
    r3[0] = np.nan
    s3[0] = np.nan
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 1)  # Need 34 for EMA34 and 1 for rolling
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if np.isnan(ema_34_1d_aligned[i]) or np.isnan(r3[i]) or np.isnan(s3[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema_1d = ema_34_1d_aligned[i]
        r3_level = r3[i]
        s3_level = s3[i]
        vol = volume[i]
        
        # Calculate 20-period volume average for spike detection
        if i >= 20:
            vol_ma = np.mean(volume[i-20:i])
        else:
            vol_ma = np.mean(volume[:i]) if i > 0 else volume[i]
        
        if position == 0:
            # Enter long: Close > R3 AND price > 1d EMA34 (uptrend) AND volume > 2x average
            if close[i] > r3_level and close[i] > ema_1d and vol > 2.0 * vol_ma:
                signals[i] = 0.25
                position = 1
            # Enter short: Close < S3 AND price < 1d EMA34 (downtrend) AND volume > 2x average
            elif close[i] < s3_level and close[i] < ema_1d and vol > 2.0 * vol_ma:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Close < S3 OR trend reverses (price < 1d EMA34)
            if close[i] < s3_level or close[i] < ema_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Close > R3 OR trend reverses (price > 1d EMA34)
            if close[i] > r3_level or close[i] > ema_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals