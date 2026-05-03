#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA200 trend filter and volume confirmation.
# In bull regime (price > 1d EMA200), go long on breakout above R3 with volume spike.
# In bear regime (price < 1d EMA200), go short on breakdown below S3 with volume spike.
# Uses Camarilla pivot levels from prior completed 1d for structure, 1d EMA200 for regime filter,
# and 4h volume spike for confirmation. Designed for 75-200 total trades over 4 years.
# Focus on BTC/ETH as primary symbols with SOL as secondary validation.

name = "4h_Camarilla_R3S3_1dEMA200_VolumeSpike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot calculation (prior completed 1d bar)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate prior 1d Camarilla pivot levels (R3, S3)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point (PP) = (H + L + C) / 3
    pp = (high_1d + low_1d + close_1d) / 3.0
    # Range = H - L
    rng = high_1d - low_1d
    # Camarilla levels
    r3 = pp + rng * 1.1 / 4.0  # R3 = PP + Range * 1.1/4
    s3 = pp - rng * 1.1 / 4.0  # S3 = PP - Range * 1.1/4
    
    # Align Camarilla levels to 4h (wait for 1d bar to complete)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Get 1d data for EMA200 trend filter
    ema_200 = pd.Series(df_1d['close'].values).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200)
    
    # Calculate volume regime: current 4h volume > 2.0x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Get current values
        close_val = close[i]
        r3_val = r3_aligned[i]
        s3_val = s3_aligned[i]
        ema_trend = ema_200_aligned[i]
        vol_spike = volume_spike[i]
        
        # Skip if any value is NaN
        if np.isnan(r3_val) or np.isnan(s3_val) or np.isnan(ema_trend):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Determine regime: bull if close > 1d EMA200, bear if close < 1d EMA200
        is_bull_regime = close_val > ema_trend
        is_bear_regime = close_val < ema_trend
        
        # Regime-based entry conditions
        if is_bull_regime:
            # Long: breakout above R3 with volume spike
            long_entry = (close_val > r3_val) and vol_spike
        else:
            long_entry = False
            
        if is_bear_regime:
            # Short: breakdown below S3 with volume spike
            short_entry = (close_val < s3_val) and vol_spike
        else:
            short_entry = False
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit on breakdown below S3 (failure of bullish breakout) or regime change to bear
            if close_val < s3_val or close_val < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit on breakout above R3 (failure of bearish breakdown) or regime change to bull
            if close_val > r3_val or close_val > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals