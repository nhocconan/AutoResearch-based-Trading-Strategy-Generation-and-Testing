# 6h_1w_camarilla_breakout_volume_v1
# Strategy: 6h Camarilla breakout with weekly trend filter and volume confirmation
# Timeframe: 6h
# Leverage: 1.0
# Hypothesis: Camarilla pivot levels (R4/S4) act as strong support/resistance. Breakouts beyond these levels
# with volume confirmation indicate institutional interest. Weekly trend filter ensures we only trade
# in the direction of the higher timeframe trend, reducing false positives. Works in both bull and bear
# markets by adapting to the weekly trend direction.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_camarilla_breakout_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly EMA(21) for trend filter
    close_1w = df_1w['close'].values
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # Daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Camarilla levels using previous day's OHLC
    # Camarilla formula: 
    # R4 = Close + ((High - Low) * 1.1 / 2)
    # R3 = Close + ((High - Low) * 1.1 / 4)
    # R2 = Close + ((High - Low) * 1.1 / 6)
    # R1 = Close + ((High - Low) * 1.1 / 12)
    # S1 = Close - ((High - Low) * 1.1 / 12)
    # S2 = Close - ((High - Low) * 1.1 / 6)
    # S3 = Close - ((High - Low) * 1.1 / 4)
    # S4 = Close - ((High - Low) * 1.1 / 2)
    
    # We need previous day's data, so we shift by 1
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Calculate Camarilla levels
    rang = prev_high - prev_low
    r4 = prev_close + (rang * 1.1 / 2)
    r3 = prev_close + (rang * 1.1 / 4)
    s3 = prev_close - (rang * 1.1 / 4)
    s4 = prev_close - (rang * 1.1 / 2)
    
    # Align Camarilla levels to 6h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_avg_20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_21_1w_aligned[i]) or np.isnan(r4_aligned[i]) or 
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(s4_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine weekly trend
        uptrend_weekly = close[i] > ema_21_1w_aligned[i]
        downtrend_weekly = close[i] < ema_21_1w_aligned[i]
        
        # Entry logic: Camarilla breakout with volume and weekly trend alignment
        # Long: break above R4 in uptrend OR break above R3 in strong uptrend
        # Short: break below S4 in downtrend OR break below S3 in strong downtrend
        if ((close[i] > r4_aligned[i] and uptrend_weekly) or 
            (close[i] > r3_aligned[i] and uptrend_weekly and vol_confirm[i])) and position != 1:
            position = 1
            signals[i] = 0.25
        elif ((close[i] < s4_aligned[i] and downtrend_weekly) or 
              (close[i] < s3_aligned[i] and downtrend_weekly and vol_confirm[i])) and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: price returns to opposite side of Camarilla bands or trend change
        elif position == 1 and (close[i] < s3_aligned[i] or not uptrend_weekly):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > r3_aligned[i] or not downtrend_weekly):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals