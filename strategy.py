#!/usr/bin/env python3
"""
6h_camarilla_pivot_1d_ema_volume_v1
Hypothesis: On 6h timeframe, use daily Camarilla pivot levels for mean-reversion entries and breakout confirmations, with daily EMA for trend filter and volume confirmation. Fade at R3/S3 levels in ranging markets, breakout continuation at R4/S4 in trending markets. Works in bull/bear via trend filter and adaptive logic.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_camarilla_pivot_1d_ema_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for Camarilla pivot and EMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    # Range
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    r4_1d = close_1d + range_1d * 1.1 / 2
    r3_1d = close_1d + range_1d * 1.1 / 4
    r2_1d = close_1d + range_1d * 1.1 / 6
    r1_1d = close_1d + range_1d * 1.1 / 12
    s1_1d = close_1d - range_1d * 1.1 / 12
    s2_1d = close_1d - range_1d * 1.1 / 6
    s3_1d = close_1d - range_1d * 1.1 / 4
    s4_1d = close_1d - range_1d * 1.1 / 2
    
    # Daily EMA for trend filter
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    
    # Align indicators to 6h timeframe
    r4_1d_6h = align_htf_to_ltf(prices, df_1d, r4_1d)
    r3_1d_6h = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_6h = align_htf_to_ltf(prices, df_1d, s3_1d)
    s4_1d_6h = align_htf_to_ltf(prices, df_1d, s4_1d)
    ema_1d_6h = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume confirmation (20-period average on 6h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(r3_1d_6h[i]) or np.isnan(s3_1d_6h[i]) or
            np.isnan(r4_1d_6h[i]) or np.isnan(s4_1d_6h[i]) or
            np.isnan(ema_1d_6h[i]) or np.isnan(vol_ma[i]) or vol_ma[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 2.0x 20-period average
        vol_confirm = volume[i] > 2.0 * vol_ma[i]
        
        # Trend direction from EMA
        uptrend = close[i] > ema_1d_6h[i]
        downtrend = close[i] < ema_1d_6h[i]
        
        if position == 1:  # Long position
            # Exit conditions
            exit_long = False
            # Exit if price reaches R4 (take profit)
            if close[i] >= r4_1d_6h[i]:
                exit_long = True
            # Exit if price crosses below S3 (stop/reversal)
            elif close[i] <= s3_1d_6h[i]:
                exit_long = True
            # Exit if trend turns down and price below pivot
            elif downtrend and close[i] < pivot_1d[i]:
                exit_long = True
            
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions
            exit_short = False
            # Exit if price reaches S4 (take profit)
            if close[i] <= s4_1d_6h[i]:
                exit_short = True
            # Exit if price crosses above R3 (stop/reversal)
            elif close[i] >= r3_1d_6h[i]:
                exit_short = True
            # Exit if trend turns up and price above pivot
            elif uptrend and close[i] > pivot_1d[i]:
                exit_short = True
            
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Determine market regime: trending or ranging
            # Using price position relative to R3/S3 as proxy
            price_vs_r3 = close[i] > r3_1d_6h[i]
            price_vs_s3 = close[i] < s3_1d_6h[i]
            
            # Long entry conditions
            long_entry = False
            if price_vs_s3 and not price_vs_r3:  # Between S3 and R3 (ranging)
                # Fade at S3: long when price touches S3 with volume confirmation
                if abs(close[i] - s3_1d_6h[i]) < (r3_1d_6h[i] - s3_1d_6h[i]) * 0.02:  # Within 2% of S3
                    if vol_confirm:
                        long_entry = True
            elif not price_vs_s3:  # Below S3 (potential breakdown)
                # Breakdown continuation: short when price breaks S4 with volume and trend
                if close[i] <= s4_1d_6h[i] and downtrend and vol_confirm:
                    # This would be a short entry, handled below
                    pass
            else:  # Above R3 (potential breakout)
                # Breakout continuation: long when price breaks R4 with volume and trend
                if close[i] >= r4_1d_6h[i] and uptrend and vol_confirm:
                    long_entry = True
            
            # Short entry conditions
            short_entry = False
            if price_vs_r3 and not price_vs_s3:  # Between S3 and R3 (ranging)
                # Fade at R3: short when price touches R3 with volume confirmation
                if abs(close[i] - r3_1d_6h[i]) < (r3_1d_6h[i] - s3_1d_6h[i]) * 0.02:  # Within 2% of R3
                    if vol_confirm:
                        short_entry = True
            elif price_vs_r3:  # Above R3 (potential breakout)
                # Breakout continuation: short when price breaks R4 with volume and trend
                if close[i] >= r4_1d_6h[i] and uptrend and vol_confirm:
                    # This would be a long entry, handled above
                    pass
            else:  # Below S3 (potential breakdown)
                # Breakdown continuation: short when price breaks S4 with volume and trend
                if close[i] <= s4_1d_6h[i] and downtrend and vol_confirm:
                    short_entry = True
            
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
    
    return signals