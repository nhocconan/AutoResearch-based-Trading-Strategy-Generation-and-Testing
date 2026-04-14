# ============================================================================
# 4h_Camilla_Signal: Camarilla Pivot Reversal with Daily Volume Spike
# ============================================================================
# Hypothesis: Camarilla pivot levels on daily timeframe provide high-probability
# reversal zones. Price retracing to these levels with volume confirmation offers
# mean-reversion entries in both bull and bear markets. Uses daily volume spike
# for confirmation and exits on opposite pivot touch. Target: 20-40 trades/year.
# ============================================================================

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop
    df_daily = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels from daily OHLC
    # Formula: R4 = C + ((H-L) * 1.1/2), R3 = C + ((H-L) * 1.1/4), etc.
    # S1 = C - ((H-L) * 1.1/6), S2 = C - ((H-L) * 1.1/4), S3 = C - ((H-L) * 1.1/2)
    daily_close = df_daily['close'].values
    daily_high = df_daily['high'].values
    daily_low = df_daily['low'].values
    
    # Calculate pivot levels
    hl_range = daily_high - daily_low
    camarilla_s3 = daily_close - (hl_range * 1.1 / 2)   # Strong support
    camarilla_s2 = daily_close - (hl_range * 1.1 / 4)   # Support
    camarilla_s1 = daily_close - (hl_range * 1.1 / 6)   # Weak support
    camarilla_r1 = daily_close + (hl_range * 1.1 / 6)   # Weak resistance
    camarilla_r2 = daily_close + (hl_range * 1.1 / 4)   # Resistance
    camarilla_r3 = daily_close + (hl_range * 1.1 / 2)   # Strong resistance
    
    # Calculate daily volume average (20-period)
    vol_daily = df_daily['volume'].values
    vol_ma_daily = pd.Series(vol_daily).rolling(window=20, min_periods=20).mean().values
    
    # Align Camarilla levels to 4h timeframe
    s3_aligned = align_htf_to_ltf(prices, df_daily, camarilla_s3)
    s2_aligned = align_htf_to_ltf(prices, df_daily, camarilla_s2)
    s1_aligned = align_htf_to_ltf(prices, df_daily, camarilla_s1)
    r1_aligned = align_htf_to_ltf(prices, df_daily, camarilla_r1)
    r2_aligned = align_htf_to_ltf(prices, df_daily, camarilla_r2)
    r3_aligned = align_htf_to_ltf(prices, df_daily, camarilla_r3)
    vol_ma_daily_aligned = align_htf_to_ltf(prices, df_daily, vol_ma_daily)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 40  # for 20-period volume MA
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(s3_aligned[i]) or np.isnan(s2_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r1_aligned[i]) or np.isnan(r2_aligned[i]) or np.isnan(r3_aligned[i]) or
            np.isnan(vol_ma_daily_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_current = volume[i]  # Current 4h volume
        
        if position == 0:
            # Long setup: price touches S3 with volume spike (strong support bounce)
            if (price <= s3_aligned[i] * 1.002 and  # Allow small buffer for wicks
                vol_current > 2.0 * vol_ma_daily_aligned[i]):  # Significant volume spike
                position = 1
                signals[i] = position_size
            # Short setup: price touches R3 with volume spike (strong resistance rejection)
            elif (price >= r3_aligned[i] * 0.998 and  # Allow small buffer for wicks
                  vol_current > 2.0 * vol_ma_daily_aligned[i]):  # Significant volume spike
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price touches R1 (weak resistance) or stops working
            if price >= r1_aligned[i] * 0.998:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price touches S1 (weak support) or stops working
            if price <= s1_aligned[i] * 1.002:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_Camilla_Signal"
timeframe = "4h"
leverage = 1.0