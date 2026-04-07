#/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 6h Camarilla Pivot Reversal with 1d EMA Trend Filter
# Hypothesis: Fade at Camarilla R3/S3 levels during strong 1d trend (EMA50) captures mean reversion in trending markets.
# Works in bull/bear: EMA50 filter ensures we only fade against the trend, avoiding counter-trend trades in strong moves.
# Target: 60-100 total trades over 4 years (15-25/year) to minimize fee drag.

name = "6h_camarilla_pivot_reversal_1d_ema_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for EMA trend and Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate EMA(50) on 1d close for trend filter
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Calculate Camarilla pivot levels from previous 1d bar
    # Using previous day's high, low, close (already available in df_1d)
    phigh = df_1d['high'].values
    plow = df_1d['low'].values
    pclose = df_1d['close'].values
    
    # Camarilla levels: R4 = C + ((H-L) * 1.1/2), R3 = C + ((H-L) * 1.1/4), etc.
    # We'll use R3/S3 for fade, R4/S4 for breakout confirmation
    range_hl = phigh - plow
    r3 = pclose + (range_hl * 1.1 / 4)
    s3 = pclose - (range_hl * 1.1 / 4)
    r4 = pclose + (range_hl * 1.1 / 2)
    s4 = pclose - (range_hl * 1.1 / 2)
    
    # Align Camarilla levels to 6h
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if required data not available
        if np.isnan(ema_50_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price reaches S3 (support) or trend changes to down
            if close[i] <= s3_aligned[i] or close[i] < ema_50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: price reaches R3 (resistance) or trend changes to up
            if close[i] >= r3_aligned[i] or close[i] > ema_50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Fade at R3/S3 in direction of 1d EMA trend
            if close[i] > ema_50_aligned[i]:  # Uptrend - look for short at R3
                if close[i] >= r3_aligned[i]:  # Price at or above R3 resistance
                    # Additional filter: avoid fading in extremely strong trends (price > R4 suggests breakout)
                    if close[i] < r4_aligned[i]:  # Below R4, not a strong breakout
                        position = -1
                        signals[i] = -0.25
            else:  # Downtrend - look for long at S3
                if close[i] <= s3_aligned[i]:  # Price at or below S3 support
                    # Additional filter: avoid fading in extremely strong trends (price < S4 suggests breakdown)
                    if close[i] > s4_aligned[i]:  # Above S4, not a strong breakdown
                        position = 1
                        signals[i] = 0.25
    
    return signals