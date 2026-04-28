#%%
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

"""
Hypothesis: 6h Williams Alligator (Jaw/Teeth/Lips) + 12h trend filter (EMA50) + volume confirmation.
The Alligator identifies trend phases: when Lips > Teeth > Jaw = uptrend, Lips < Teeth < Jaw = downtrend.
We enter only when aligned with higher timeframe trend and volume confirms momentum.
Designed to work in both bull (trend following) and bear (avoiding false signals via 12h filter).
Target: 20-60 trades/year on 6f timeframe.
"""

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Williams Alligator (triggers)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 13:
        return np.zeros(n)
    
    # Williams Alligator: Jaw (13,8), Teeth (8,5), Lips (5,3) - all SMMA
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    median_12h = (high_12h + low_12h) / 2
    
    def smma(series, period):
        if len(series) < period:
            return np.full_like(series, np.nan, dtype=np.float64)
        result = np.full_like(series, np.nan, dtype=np.float64)
        # First value is SMA
        result[period-1] = np.mean(series[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT) / period
        for i in range(period, len(series)):
            result[i] = (result[i-1] * (period-1) + series[i]) / period
        return result
    
    jaw = smma(median_12h, 13)  # Blue line
    teeth = smma(median_12h, 8)  # Red line
    lips = smma(median_12h, 5)   # Green line
    
    # Align Alligator lines to 6h timeframe (wait for 12h bar close)
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips)
    
    # Get 1d data for trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume filter: above 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 8-20 UTC (most active trading hours)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 8-20 UTC
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            # Outside session: flatten position
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Volume filter: above average volume
        vol_filter = volume[i] > vol_ma[i]
        
        # Trend filter: price relative to 1d EMA50
        trend_up = close[i] > ema50_1d_aligned[i]
        trend_down = close[i] < ema50_1d_aligned[i]
        
        # Alligator alignment signals
        # Uptrend: Lips > Teeth > Jaw (green > red > blue)
        alligator_up = lips_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > jaw_aligned[i]
        # Downtrend: Lips < Teeth < Jaw (green < red < blue)
        alligator_down = lips_aligned[i] < teeth_aligned[i] and teeth_aligned[i] < jaw_aligned[i]
        
        # Entry conditions: Alligator alignment + trend + volume
        long_entry = alligator_up and trend_up and vol_filter
        short_entry = alligator_down and trend_down and vol_filter
        
        # Exit conditions: opposite Alligator alignment (trend change)
        long_exit = alligator_down and position == 1
        short_exit = alligator_up and position == -1
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit:
            signals[i] = 0.0
            position = 0
        elif short_exit:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_WilliamsAlligator_1dEMA50_Volume"
timeframe = "6h"
leverage = 1.0
#%%