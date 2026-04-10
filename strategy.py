#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout with 12h volume confirmation and 1d trend filter
# - Primary signal: Price breaks above R4 or below S4 Camarilla levels calculated from prior 1d session
# - Volume confirmation: 12h volume > 1.3x 20-period average volume (avoid false breakouts)
# - Trend filter: 1d close > EMA(50) for longs, close < EMA(50) for shorts (align with higher TF trend)
# - Works in bull/bear: In uptrends, R4 breakouts continue momentum; in downtrends, S4 breaks extend decline
# - Position size: 0.25 discrete level to minimize fee churn
# - Target: 12-30 trades/year (50-120 total over 4 years) per 6h strategy guidelines
# - No stoploss: exit on opposite Camarilla level touch (R3/S3) or reverse signal

name = "6h_12h_1d_camarilla_breakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_12h = get_htf_data(prices, '12h')
    if len(df_1d) < 50 or len(df_12h) < 20:
        return np.zeros(n)
    
    # Pre-compute 1d Camarilla levels from prior day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: based on previous day's range
    R4 = close_1d + ((high_1d - low_1d) * 1.1 / 2)
    R3 = close_1d + ((high_1d - low_1d) * 1.1 / 4)
    R2 = close_1d + ((high_1d - low_1d) * 1.1 / 6)
    R1 = close_1d + ((high_1d - low_1d) * 1.1 / 12)
    S1 = close_1d - ((high_1d - low_1d) * 1.1 / 12)
    S2 = close_1d - ((high_1d - low_1d) * 1.1 / 6)
    S3 = close_1d - ((high_1d - low_1d) * 1.1 / 4)
    S4 = close_1d - ((high_1d - low_1d) * 1.1 / 2)
    
    # Align Camarilla levels to 6h timeframe (use prior day's levels)
    R4_aligned = align_htf_to_ltf(prices, df_1d, R4)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    S4_aligned = align_htf_to_ltf(prices, df_1d, S4)
    
    # Pre-compute 12h volume spike filter
    volume_12h = df_12h['volume'].values
    avg_volume_20 = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_12h > (1.3 * avg_volume_20)
    volume_spike_aligned = align_htf_to_ltf(prices, df_12h, volume_spike)
    
    # Pre-compute 1d EMA(50) for trend filter
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # 6h price data
    close_6h = prices['close'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(R4_aligned[i]) or np.isnan(R3_aligned[i]) or
            np.isnan(S3_aligned[i]) or np.isnan(S4_aligned[i]) or
            np.isnan(volume_spike_aligned[i]) or np.isnan(ema_50_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price touches R3 (take profit) or reverse signal
            if close_6h[i] <= R3_aligned[i]:
                position = 0
                signals[i] = 0.0
            elif close_6h[i] < S4_aligned[i] and volume_spike_aligned[i]:
                # Reverse to short if breaks S4 with volume
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price touches S3 (take profit) or reverse signal
            if close_6h[i] >= S3_aligned[i]:
                position = 0
                signals[i] = 0.0
            elif close_6h[i] > R4_aligned[i] and volume_spike_aligned[i]:
                # Reverse to long if breaks R4 with volume
                position = 1
                signals[i] = 0.25
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Camarilla breakouts with volume confirmation and trend filter
            # Long: price breaks above R4 with volume and uptrend
            if (close_6h[i] > R4_aligned[i] and 
                volume_spike_aligned[i] and 
                close_6h[i] > ema_50_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short: price breaks below S4 with volume and downtrend
            elif (close_6h[i] < S4_aligned[i] and 
                  volume_spike_aligned[i] and 
                  close_6h[i] < ema_50_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals