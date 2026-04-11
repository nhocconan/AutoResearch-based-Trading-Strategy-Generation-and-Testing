#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h_1d_camarilla_breakout_v1"
timeframe = "1h"
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
    
    # Load 4h and 1d data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 50 or len(df_1d) < 50:
        return signals
    
    # Calculate 4h ATR for volatility filter
    tr1 = df_4h['high'] - df_4h['low']
    tr2 = np.abs(df_4h['high'] - df_4h['close'].shift(1))
    tr3 = np.abs(df_4h['low'] - df_4h['close'].shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_4h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate daily Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    daily_range = high_1d - low_1d
    r4 = close_1d + (daily_range * 1.1 / 2)  # Resistance level 4
    s4 = close_1d - (daily_range * 1.1 / 2)  # Support level 4
    r3 = close_1d + (daily_range * 1.1 / 4)  # Resistance level 3
    s3 = close_1d - (daily_range * 1.1 / 4)  # Support level 3
    
    # Align 4h ATR and daily levels to 1h timeframe
    atr_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_4h)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # Volume filter: 1h volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(atr_4h_aligned[i]) or np.isnan(r4_aligned[i]) or 
            np.isnan(s4_aligned[i]) or np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 08-20 UTC
        if not (8 <= hours[i] <= 20):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        volume_current = volume[i]
        
        # Volatility filter: require minimum 4h ATR
        vol_filter = atr_4h_aligned[i] > 0.001 * price_close  # at least 0.1% of price
        
        # Volume confirmation
        vol_confirm = volume_current > 1.5 * vol_ma_20[i]
        
        # Breakout conditions
        breakout_up = price_close > r4_aligned[i]
        breakout_down = price_close < s4_aligned[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Break above R4 with volume and volatility confirmation
        if breakout_up and vol_confirm and vol_filter:
            enter_long = True
        
        # Short: Break below S4 with volume and volatility confirmation
        if breakout_down and vol_confirm and vol_filter:
            enter_short = True
        
        # Exit conditions: return to opposite S3/R3 levels
        exit_long = price_close < s3_aligned[i]
        exit_short = price_close > r3_aligned[i]
        
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

# Hypothesis: 1h Camarilla breakout strategy using 4h volatility filter and 1d pivot levels.
# Enters long when price breaks above R4 with volume > 1.5x 20-period average and sufficient 4h volatility.
# Enters short when price breaks below S4 with same conditions.
# Uses session filter (08-20 UTC) to avoid low-volume Asian session noise.
# Position size fixed at 0.20 to minimize risk and allow for multiple entries.
# Target: 15-25 trades per year (60-100 total over 4 years) to stay under fee drag threshold.
# Works in both bull and bear markets by capturing significant breakouts in either direction.
# Uses 4h ATR to avoid choppy markets and 1d Camarilla levels for institutional reference points.