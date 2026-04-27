# [EXPERIMENT #96367] 6h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_v2
# Hypothesis: 6h Camarilla R3/S3 breakouts with 1d EMA34 trend filter and volume spike capture
# institutional breakouts with low frequency (target 15-30/year). Works in bull/bear via trend filter.
# Uses discrete sizing (0.25) to minimize fee churn. Avoids overtrading via strict confluence.

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
    
    # Camarilla levels from prior day (using typical price)
    typical_price = (high + low + close) / 3
    # Use prior day's typical price for Camarilla calculation
    # Shift by 1 to use completed day's data
    typical_price_shifted = np.roll(typical_price, 1)
    typical_price_shifted[0] = np.nan  # First value invalid
    
    # Calculate Camarilla levels for each bar based on prior day's action
    # We'll compute daily Camarilla and align
    # For simplicity, we use rolling window to approximate prior day's range
    # In practice, we'd use actual daily OHLC, but we approximate with 24-period lookback (6h * 4 = 24h)
    # Better: use actual 1d data for Camarilla calculation
    
    # Get 1d data for proper Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: based on previous day's range
    # R4 = C + (H-L)*1.1/2
    # R3 = C + (H-L)*1.1/4
    # R2 = C + (H-L)*1.1/6
    # R1 = C + (H-L)*1.1/12
    # PP = (H+L+C)/3
    # S1 = C - (H-L)*1.1/12
    # S2 = C - (H-L)*1.1/6
    # S3 = C - (H-L)*1.1/4
    # S4 = C - (H-L)*1.1/2
    
    # Calculate for each day using prior day's OHLC
    # Shift 1 to use completed day
    high_1d_prev = np.roll(high_1d, 1)
    low_1d_prev = np.roll(low_1d, 1)
    close_1d_prev = np.roll(close_1d, 1)
    # First day values invalid
    high_1d_prev[0] = np.nan
    low_1d_prev[0] = np.nan
    close_1d_prev[0] = np.nan
    
    # Camarilla calculations
    H = high_1d_prev
    L = low_1d_prev
    C = close_1d_prev
    range_hl = H - L
    
    R4 = C + range_hl * 1.1 / 2
    R3 = C + range_hl * 1.1 / 4
    S3 = C - range_hl * 1.1 / 4
    S4 = C - range_hl * 1.1 / 2
    
    # 1d EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Volume confirmation: current 6h volume > 2.0 * 24-period average (approx 1 day)
    vol_avg = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_confirm = volume > (2.0 * vol_avg)
    
    # Align all 1d indicators to 6h timeframe
    R3_6h = align_htf_to_ltf(prices, df_1d, R3)
    S3_6h = align_htf_to_ltf(prices, df_1d, S3)
    R4_6h = align_htf_to_ltf(prices, df_1d, R4)
    S4_6h = align_htf_to_ltf(prices, df_1d, S4)
    ema34_1d_6h = align_htf_to_ltf(prices, df_1d, ema34_1d)
    volume_confirm_6h = align_htf_to_ltf(prices, df_1d, volume_confirm)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need Camarilla (needs 2 days), EMA34 (34), volume avg (24)
    start_idx = max(48, 34, 24)  # 2 days for Camarilla (48 bars of 6h)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(R3_6h[i]) or np.isnan(S3_6h[i]) or 
            np.isnan(R4_6h[i]) or np.isnan(S4_6h[i]) or
            np.isnan(ema34_1d_6h[i]) or np.isnan(volume_confirm_6h[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        r3 = R3_6h[i]
        s3 = S3_6h[i]
        r4 = R4_6h[i]
        s4 = S4_6h[i]
        ema34 = ema34_1d_6h[i]
        vol_conf = volume_confirm_6h[i]
        
        if position == 0:
            # Determine trend: price vs EMA34 (1d)
            uptrend = close_val > ema34
            downtrend = close_val < ema34
            
            # Long: break above R3 with volume in uptrend
            if uptrend and vol_conf and close_val > r3:
                signals[i] = size
                position = 1
            # Short: break below S3 with volume in downtrend
            elif downtrend and vol_conf and close_val < s3:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Long exit: price re-enters below R3 or trend reversal
            if close_val < r3:  # Re-enter below R3
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: price re-enters above S3 or trend reversal
            if close_val > s3:  # Re-enter above S3
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_v2"
timeframe = "6h"
leverage = 1.0