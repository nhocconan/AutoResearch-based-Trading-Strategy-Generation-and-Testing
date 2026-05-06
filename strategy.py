#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using daily Camarilla pivot levels with volume spike and trend filter
# - Uses 1d Camarilla pivot levels (R3, R4, S3, S4) for mean reversion and breakout signals
# - Uses 6h volume spike for entry confirmation
# - Uses 6h ADX > 20 to filter for trending markets only
# - Enters long when price breaks above R4 with volume and trend (breakout continuation)
# - Enters short when price breaks below S4 with volume and trend (breakout continuation)
# - Enters long when price bounces off S3 with volume and trend (mean reversion in uptrend)
# - Enters short when price bounces off R3 with volume and trend (mean reversion in downtrend)
# - Exits when price reaches opposite Camarilla level (R3/S3 for mean reversion, R4/S4 for breakout)
# - Designed to capture both breakout and mean reversion opportunities in trending markets
# - Target: 60-120 total trades over 4 years (15-30/year) with 0.25 position sizing

name = "6h_1dCamarilla_R3R4_S3S4_Volume_ADX"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's values for pivot calculation
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = high_1d[0]  # First day uses current day's high
    prev_low[0] = low_1d[0]    # First day uses current day's low
    prev_close[0] = close_1d[0] # First day uses current day's close
    
    # Camarilla pivot calculation
    pivot = (prev_high + prev_low + prev_close) / 3
    range_ = prev_high - prev_low
    
    # Resistance levels
    R3 = pivot + (range_ * 1.1 / 2)
    R4 = pivot + (range_ * 1.1)
    
    # Support levels
    S3 = pivot - (range_ * 1.1 / 2)
    S4 = pivot - (range_ * 1.1)
    
    # Align 1d Camarilla levels to 6h timeframe
    R3_6h = align_htf_to_ltf(prices, df_1d, R3)
    R4_6h = align_htf_to_ltf(prices, df_1d, R4)
    S3_6h = align_htf_to_ltf(prices, df_1d, S3)
    S4_6h = align_htf_to_ltf(prices, df_1d, S4)
    
    # Volume filter (6h timeframe)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)  # Strong volume confirmation
    
    # ADX filter (6h timeframe) - trend strength
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            plus_dm[i] = max(high[i] - high[i-1], 0)
            minus_dm[i] = max(low[i-1] - low[i], 0)
            if plus_dm[i] < minus_dm[i]:
                plus_dm[i] = 0
            if minus_dm[i] < plus_dm[i]:
                minus_dm[i] = 0
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Smooth with Wilder's smoothing (alpha = 1/period)
        atr = np.zeros_like(high)
        plus_di = np.zeros_like(high)
        minus_di = np.zeros_like(high)
        
        atr[period-1] = np.mean(tr[1:period+1])
        plus_dm_sum = np.sum(plus_dm[1:period+1])
        minus_dm_sum = np.sum(minus_dm[1:period+1])
        
        for i in range(period, len(high)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
            plus_dm_sum = plus_dm_sum - (plus_dm[i-period+1] if i-period+1 >= 0 else 0) + plus_dm[i]
            minus_dm_sum = minus_dm_sum - (minus_dm[i-period+1] if i-period+1 >= 0 else 0) + minus_dm[i]
            plus_di[i] = 100 * plus_dm_sum / (atr[i] * period) if atr[i] != 0 else 0
            minus_di[i] = 100 * minus_dm_sum / (atr[i] * period) if atr[i] != 0 else 0
        
        dx = np.zeros_like(high)
        adx = np.zeros_like(high)
        for i in range(2*period-1, len(high)):
            di_diff = abs(plus_di[i] - minus_di[i])
            di_sum = plus_di[i] + minus_di[i]
            dx[i] = 100 * di_diff / di_sum if di_sum != 0 else 0
        
        # Smooth DX to get ADX
        adx[2*period-1] = np.mean(dx[2*period-1:3*period]) if 3*period <= len(high) else 0
        for i in range(3*period, len(high)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx_values = calculate_adx(high, low, close, 14)
    adx_filter = adx_values > 20  # Trend filter (lower threshold for more signals)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN
        if (np.isnan(R3_6h[i]) or np.isnan(R4_6h[i]) or 
            np.isnan(S3_6h[i]) or np.isnan(S4_6h[i]) or 
            np.isnan(volume_spike[i]) or np.isnan(adx_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price breaks above R4 with volume and trend
            if close[i] > R4_6h[i] and volume_spike[i] and adx_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short breakout: price breaks below S4 with volume and trend
            elif close[i] < S4_6h[i] and volume_spike[i] and adx_filter[i]:
                signals[i] = -0.25
                position = -1
            # Long mean reversion: price bounces off S3 with volume and trend (in uptrend)
            elif close[i] > S3_6h[i] and close[i-1] <= S3_6h[i-1] and volume_spike[i] and adx_filter[i]:
                # Additional check for uptrend: price above pivot
                if close[i] > pivot[i] if not np.isnan(pivot[i]) else False:
                    signals[i] = 0.25
                    position = 1
            # Short mean reversion: price bounces off R3 with volume and trend (in downtrend)
            elif close[i] < R3_6h[i] and close[i-1] >= R3_6h[i-1] and volume_spike[i] and adx_filter[i]:
                # Additional check for downtrend: price below pivot
                if close[i] < pivot[i] if not np.isnan(pivot[i]) else False:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: 
            # For breakout: price returns to R3
            # For mean reversion: price reaches R4
            if close[i] < R3_6h[i] or close[i] > R4_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: 
            # For breakout: price returns to S3
            # For mean reversion: price reaches S4
            if close[i] > S3_6h[i] or close[i] < S4_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals