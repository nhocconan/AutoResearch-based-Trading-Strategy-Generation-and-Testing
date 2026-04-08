# 6h_1d_camarilla_vol_breakout
# Hypothesis: Fade at Camarilla R3/S3, breakout continuation at R4/S4 on 6h with daily trend filter and volume confirmation.
# In ranging markets, price reverts from extreme Camarilla levels (R3/S3). In trending markets, breaks of R4/S4
# with volume and daily trend alignment continue the move. Works in both bull/bear via adaptive logic.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_camarilla_vol_breakout"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily data for Camarilla levels and trend
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels from previous day
    # Camarilla: R4 = C + (H-L)*1.1/2, R3 = C + (H-L)*1.1/4, etc.
    # Using previous day's OHLC
    prev_close = np.roll(close_1d, 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    
    # Avoid division by zero in range
    prev_range = prev_high - prev_low
    prev_range = np.where(prev_range == 0, 1e-10, prev_range)
    
    # Camarilla levels
    r4 = prev_close + prev_range * 1.1 / 2
    r3 = prev_close + prev_range * 1.1 / 4
    s3 = prev_close - prev_range * 1.1 / 4
    s4 = prev_close - prev_range * 1.1 / 2
    
    # Daily EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all to 6h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: volume > 1.8x 24-period average (~4 days)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 50  # Ensure EMA50 is ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r4_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(ema50_aligned[i]) or np.isnan(vol_ma_24[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Volume surge condition
        vol_surge = volume[i] > 1.8 * vol_ma_24[i] if vol_ma_24[i] > 0 else False
        
        if position == 1:  # Long position
            # Exit: price breaks below S3 (mean reversion failure) or reverses at R4
            if close[i] < s3_aligned[i]:
                position = 0
                signals[i] = 0.0
            elif close[i] > r4_aligned[i] and vol_surge:
                # Breakout continuation - hold
                signals[i] = 0.25
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above R3 (mean reversion failure) or reverses at S4
            if close[i] > r3_aligned[i]:
                position = 0
                signals[i] = 0.0
            elif close[i] < s4_aligned[i] and vol_surge:
                # Breakout continuation - hold
                signals[i] = -0.25
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Determine market regime: trend vs range based on price vs EMA50
            price_above_ema = close[i] > ema50_aligned[i]
            price_below_ema = close[i] < ema50_aligned[i]
            
            # In uptrend (price > EMA50): look for longs at S3/S4, shorts on R4 break
            if price_above_ema:
                # Long: mean reversion from S3/S4
                if close[i] <= s3_aligned[i] and vol_surge:
                    position = 1
                    signals[i] = 0.25
                # Short: breakout continuation below S4
                elif close[i] < s4_aligned[i] and vol_surge:
                    position = -1
                    signals[i] = -0.25
            # In downtrend (price < EMA50): look for shorts at R3/R4, longs on R3 break
            elif price_below_ema:
                # Short: mean reversion from R3/R4
                if close[i] >= r3_aligned[i] and vol_surge:
                    position = -1
                    signals[i] = -0.25
                # Long: breakout continuation above R4
                elif close[i] > r4_aligned[i] and vol_surge:
                    position = 1
                    signals[i] = 0.25
            # In transition (price near EMA): fade extremes
            else:
                # Long at S3 (oversold bounce)
                if close[i] <= s3_aligned[i] and vol_surge:
                    position = 1
                    signals[i] = 0.25
                # Short at R3 (overbought reversal)
                elif close[i] >= r3_aligned[i] and vol_surge:
                    position = -1
                    signals[i] = -0.25
    
    return signals