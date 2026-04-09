#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout with 1d volume regime filter
# - Primary signal: 6h price breaks above Camarilla R4 or below S4 (strong breakout)
# - Trend filter: 1d volume > 20-period median volume (avoid low-participation breakouts)
# - Entry only in direction of 1d price vs VWAP (long if price > VWAP, short if price < VWAP)
# - Position size: 0.25 (discrete level) to minimize fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) per 6h strategy guidelines
# - Works in bull/bear: Camarilla breakouts capture strong moves, volume filter ensures participation,
#   VWAP filter aligns with higher timeframe trend, reducing false breakouts in chop

name = "6h_1d_camarilla_breakout_volume_vwap_v1"
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
    volume_1d = df_1d['volume'].values
    
    # 1d VWAP (volume weighted average price)
    typical_price_1d = (high_1d + low_1d + close_1d) / 3.0
    vwap_num = np.cumsum(typical_price_1d * volume_1d)
    vwap_den = np.cumsum(volume_1d)
    vwap_1d = np.where(vwap_den != 0, vwap_num / vwap_den, typical_price_1d)
    
    # Align 1d VWAP to 6h timeframe (completed 1d bar only)
    vwap_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    
    # 1d volume regime: volume > 20-period median volume
    median_volume_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).median().values
    volume_regime = volume_1d > median_volume_20
    volume_regime_aligned = align_htf_to_ltf(prices, df_1d, volume_regime)
    
    # 6h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate Camarilla levels for 6h using previous 6h bar's OHLC
    # Camarilla: based on previous day's range, but we apply to previous 6h bar
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close[0] = close[0]  # avoid NaN on first bar
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    
    # Camarilla levels
    range_ = prev_high - prev_low
    camarilla_r4 = prev_close + range_ * 1.1 / 2  # R4 = C + (H-L)*1.1/2
    camarilla_s4 = prev_close - range_ * 1.1 / 2  # S4 = C - (H-L)*1.1/2
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(vwap_aligned[i]) or
            np.isnan(volume_regime_aligned[i]) or
            np.isnan(camarilla_r4[i]) or
            np.isnan(camarilla_s4[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below VWAP OR Camarilla S3 (profit taking)
            camarilla_s3 = prev_close[i-1] - (prev_high[i-1] - prev_low[i-1]) * 1.1 / 4 if i >= 1 else camarilla_s4[i]
            if close[i] < vwap_aligned[i] or close[i] < camarilla_s3:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above VWAP OR Camarilla R3 (profit taking)
            camarilla_r3 = prev_close[i-1] + (prev_high[i-1] - prev_low[i-1]) * 1.1 / 4 if i >= 1 else camarilla_r4[i]
            if close[i] > vwap_aligned[i] or close[i] > camarilla_r3:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Camarilla breakouts with volume regime and VWAP filter
            # Breakout long: price > R4 AND volume regime AND price > VWAP
            if close[i] > camarilla_r4[i] and volume_regime_aligned[i] and close[i] > vwap_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Breakout short: price < S4 AND volume regime AND price < VWAP
            elif close[i] < camarilla_s4[i] and volume_regime_aligned[i] and close[i] < vwap_aligned[i]:
                position = -1
                signals[i] = -0.25
    
    return signals