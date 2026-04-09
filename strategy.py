#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout with 1d ADX trend filter and volume confirmation
# - Uses 1d Camarilla pivot levels (R3/S3 for mean reversion, R4/S4 for breakout)
# - Breakout long when price > R4 and 1d ADX > 25 (strong trend)
# - Breakout short when price < S4 and 1d ADX > 25 (strong trend)
# - Mean reversion long when price < S3 and 1d ADX < 20 (range market)
# - Mean reversion short when price > R3 and 1d ADX < 20 (range market)
# - Volume > 1.5x 20-period average confirms momentum
# - Position size: 0.25 (25% of capital) - discrete level to minimize fee churn
# - Target: ~20 trades/year (80 total over 4 years) to stay well under fee drag threshold
# - Camarilla pivots work well in both trending and ranging markets when combined with ADX regime filter
# - Designed to capture strong breaks in trends and fade extremes in ranges

name = "6h_1d_camarilla_adx_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 1d indicators
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d Camarilla pivot levels (based on previous day)
    # Camarilla: R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low)
    #           S3 = close - 1.1*(high-low), S4 = close - 1.5*(high-low)
    camarilla_high = np.maximum(high_1d, low_1d)  # ensure valid
    camarilla_low = np.minimum(high_1d, low_1d)
    camarilla_range = camarilla_high - camarilla_low
    camarilla_range = np.where(camarilla_range <= 0, np.full_like(camarilla_range, 1e-8), camarilla_range)
    
    r4 = close_1d + 1.5 * camarilla_range
    r3 = close_1d + 1.1 * camarilla_range
    s3 = close_1d - 1.1 * camarilla_range
    s4 = close_1d - 1.5 * camarilla_range
    
    # 1d ADX(14) for trend filter
    # ADX calculation: +DM, -DM, TR, then DX, then ADX
    high_shift = np.roll(high_1d, 1)
    low_shift = np.roll(low_1d, 1)
    close_shift = np.roll(close_1d, 1)
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - close_shift)
    tr3 = np.abs(low_1d - close_shift)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    plus_dm = np.where((high_1d - high_shift) > (low_shift - low_1d), np.maximum(high_1d - high_shift, 0), 0)
    minus_dm = np.where((low_shift - low_1d) > (high_1d - high_shift), np.maximum(low_shift - low_1d, 0), 0)
    
    # Smoothed values (Wilder's smoothing)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(data[:period])
        # Subsequent values: prev*(period-1) + current) / period
        for i in range(period, len(data)):
            if not np.isnan(result[i-1]):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    period = 14
    atr_1d = wilders_smoothing(tr, period)
    plus_di_1d = 100 * wilders_smoothing(plus_dm, period) / atr_1d
    minus_di_1d = 100 * wilders_smoothing(minus_dm, period) / atr_1d
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)
    adx_1d = wilders_smoothing(dx_1d, period)
    
    # Align HTF indicators to LTF
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # 6h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 6h volume > 1.5x 20-period average for confirmation
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(r4_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(adx_1d_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit conditions: mean reversion in range OR trend exhaustion
            if adx_1d_aligned[i] < 20 and close[i] > s3_aligned[i]:  # range: exit long at S3
                position = 0
                signals[i] = 0.0
            elif adx_1d_aligned[i] >= 20 and close[i] < r3_aligned[i]:  # trend: exit at R3
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions: mean reversion in range OR trend exhaustion
            if adx_1d_aligned[i] < 20 and close[i] < r3_aligned[i]:  # range: exit short at R3
                position = 0
                signals[i] = 0.0
            elif adx_1d_aligned[i] >= 20 and close[i] > s3_aligned[i]:  # trend: exit at S3
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for breakout or mean reversion opportunities
            # Breakout long: price > R4 AND strong trend (ADX > 25) AND volume spike
            if (close[i] > r4_aligned[i] and 
                adx_1d_aligned[i] > 25 and
                volume_spike[i]):
                position = 1
                signals[i] = 0.25
            # Breakout short: price < S4 AND strong trend (ADX > 25) AND volume spike
            elif (close[i] < s4_aligned[i] and 
                  adx_1d_aligned[i] > 25 and
                  volume_spike[i]):
                position = -1
                signals[i] = -0.25
            # Mean reversion long: price < S3 AND ranging market (ADX < 20) AND volume spike
            elif (close[i] < s3_aligned[i] and 
                  adx_1d_aligned[i] < 20 and
                  volume_spike[i]):
                position = 1
                signals[i] = 0.25
            # Mean reversion short: price > R3 AND ranging market (ADX < 20) AND volume spike
            elif (close[i] > r3_aligned[i] and 
                  adx_1d_aligned[i] < 20 and
                  volume_spike[i]):
                position = -1
                signals[i] = -0.25
    
    return signals