#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation
# Camarilla pivot levels provide high-probability reversal/breakout points in intraday trading.
# R3 and S3 levels represent strong resistance/support where price often accelerates after breaking.
# Combined with 1d EMA34 trend filter to ensure alignment with higher timeframe direction.
# Volume spike confirms institutional participation. Works in bull/bear via trend filter.
# Target: 75-200 total trades over 4 years (19-50/year) with discrete sizing 0.25.

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_Trend_Volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate EMA(34) on 1d for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla pivot levels from previous day
    # Typical Camarilla formula based on previous day's OHLC
    # We'll use the 1d data to calculate levels for the current 4h session
    # For each 4h bar, we need the previous 1d bar's OHLC
    
    # Shift 1d data by 1 to get previous day's values (avoid look-ahead)
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Calculate Camarilla levels
    # R3 = Close + (High - Low) * 1.1/4
    # S3 = Close - (High - Low) * 1.1/4
    diff = prev_high - prev_low
    r3 = prev_close + diff * 1.1 / 4
    s3 = prev_close - diff * 1.1 / 4
    
    # Align to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume confirmation (2.0x 20-period average) on 4h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for calculations)
    start_idx = 20
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Close breaks above R3 + 1d uptrend + volume spike
            if close[i] > r3_aligned[i] and close[i-1] <= r3_aligned[i-1] and close[i] > ema_34_1d_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short entry: Close breaks below S3 + 1d downtrend + volume spike
            elif close[i] < s3_aligned[i] and close[i-1] >= s3_aligned[i-1] and close[i] < ema_34_1d_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Close breaks below R3 or trend reversal
            if close[i] < r3_aligned[i] and close[i-1] >= r3_aligned[i-1] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Close breaks above S3 or trend reversal
            if close[i] > s3_aligned[i] and close[i-1] <= s3_aligned[i-1] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals