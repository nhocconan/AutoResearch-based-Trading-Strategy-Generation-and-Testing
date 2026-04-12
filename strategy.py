# 1h_4h_1d_camel_breakout_v2
# Strategy: 1h chart with 4h/1d Camarilla pivot + volume spike for breakout entries.
# Uses higher timeframes for direction (4h trend, 1d bias) and 1h for precise entry timing.
# Volume filter ensures momentum confirmation. Session filter (08-20 UTC) reduces noise.
# Target: 20-40 trades/year per symbol to minimize fee drag in choppy 1h environment.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for trend and Camarilla levels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Get 1d data for bias
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # 4h EMA20 for trend direction
    ema20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # 1d close vs 20-period EMA for bias
    ema20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Camarilla levels for 4h (based on previous 4h bar)
    # Calculate using typical price of previous completed 4h bar
    typical_4h = (high_4h + low_4h + close_4h) / 3
    S1 = typical_4h - 1.1 * (high_4h - low_4h) / 12
    S2 = typical_4h - 1.1 * (high_4h - low_4h) / 6
    S3 = typical_4h - 1.1 * (high_4h - low_4h) / 4
    R1 = typical_4h + 1.1 * (high_4h - low_4h) / 12
    R2 = typical_4h + 1.1 * (high_4h - low_4h) / 6
    R3 = typical_4h + 1.1 * (high_4h - low_4h) / 4
    
    # Align all 4h/1d data to 1h
    ema20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema20_4h)
    ema20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema20_1d)
    S1_aligned = align_htf_to_ltf(prices, df_4h, S1)
    S2_aligned = align_htf_to_ltf(prices, df_4h, S2)
    S3_aligned = align_htf_to_ltf(prices, df_4h, S3)
    R1_aligned = align_htf_to_ltf(prices, df_4h, R1)
    R2_aligned = align_htf_to_ltf(prices, df_4h, R2)
    R3_aligned = align_htf_to_ltf(prices, df_4h, R3)
    
    # 1h volume spike detection (volume > 1.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    
    for i in range(30, n):
        # Skip if not in session or data not ready
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        if (np.isnan(ema20_4h_aligned[i]) or np.isnan(ema20_1d_aligned[i]) or
            np.isnan(S1_aligned[i]) or np.isnan(R1_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Determine bias from 1d
        bullish_bias = close_1d[-1] > ema20_1d[-1] if len(close_1d) > 0 else False
        bearish_bias = close_1d[-1] < ema20_1d[-1] if len(close_1d) > 0 else False
        
        # Determine 4h trend
        uptrend_4h = ema20_4h_aligned[i] > ema20_4h_aligned[i-1]
        downtrend_4h = ema20_4h_aligned[i] < ema20_4h_aligned[i-1]
        
        # Long conditions: bullish bias + uptrend + break above R1 with volume
        if bullish_bias and uptrend_4h:
            if close[i] > R1_aligned[i] and vol_spike[i]:
                signals[i] = 0.20
            else:
                signals[i] = 0.0
        # Short conditions: bearish bias + downtrend + break below S1 with volume
        elif bearish_bias and downtrend_4h:
            if close[i] < S1_aligned[i] and vol_spike[i]:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals

name = "1h_4h_1d_camel_breakout_v2"
timeframe = "1h"
leverage = 1.0