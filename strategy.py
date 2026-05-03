#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla R3/S3 breakout with 1w EMA34 trend filter and volume spike confirmation.
# In bull regime (price > 1w EMA34), go long on breakout above R3 with volume spike.
# In bear regime (price < 1w EMA34), go short on breakdown below S3 with volume spike.
# Uses Camarilla pivot levels from prior completed 1w for structure, 1w EMA34 for regime filter,
# and 1d volume spike (>2.0x 20-period MA) for confirmation. Designed for 30-100 total trades over 4 years.
# Focus on BTC/ETH as primary symbols.

name = "1d_Camarilla_R3S3_1wEMA34_VolumeSpike"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for Camarilla pivot calculation (prior completed 1w bar)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate prior 1w Camarilla levels
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Pivot point
    pivot = (high_1w + low_1w + close_1w) / 3.0
    # Range
    rng = high_1w - low_1w
    # Camarilla levels
    r3 = pivot + rng * 1.1 / 4.0
    s3 = pivot - rng * 1.1 / 4.0
    
    # Align Camarilla levels to 1d (wait for 1w bar to complete)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    
    # Get 1w data for EMA34 trend filter
    ema_34 = pd.Series(df_1w['close'].values).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1w, ema_34)
    
    # Calculate volume regime: current 1d volume > 2.0x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Get current values
        close_val = close[i]
        r3_val = r3_aligned[i]
        s3_val = s3_aligned[i]
        ema_trend = ema_34_aligned[i]
        vol_spike = volume_spike[i]
        
        # Skip if any value is NaN
        if np.isnan(r3_val) or np.isnan(s3_val) or np.isnan(ema_trend):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Determine regime: bull if close > 1w EMA34, bear if close < 1w EMA34
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