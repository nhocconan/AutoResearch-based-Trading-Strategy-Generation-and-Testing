# 2025-06-20: 4h volume-weighted VWAP deviation + 1d ADX trend filter
# Hypothesis: In ranging markets, price reverts to VWAP; in trending markets, 
# price continues in trend direction. Combines mean reversion and trend following
# with volume confirmation. Target: 20-40 trades/year per symbol.
# Works in bull (trend continuation) and bear (mean reversion in ranges).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_vwap(high, low, close, volume, window):
    """Calculate Volume Weighted Average Price over rolling window."""
    typical_price = (high + low + close) / 3.0
    vwap_numerator = pd.Series(typical_price * volume).rolling(window=window, min_periods=1).sum()
    vwap_denominator = pd.Series(volume).rolling(window=window, min_periods=1).sum()
    vwap = vwap_numerator / vwap_denominator
    return vwap.values

def calculate_adx(high, low, close, period=14):
    """Calculate Average Directional Index."""
    plus_dm = np.zeros(len(high))
    minus_dm = np.zeros(len(high))
    tr = np.zeros(len(high))
    
    for i in range(1, len(high)):
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]
        
        plus_dm[i] = high_diff if high_diff > low_diff and high_diff > 0 else 0
        minus_dm[i] = low_diff if low_diff > high_diff and low_diff > 0 else 0
        
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]),
            abs(low[i] - close[i-1])
        )
    
    # Wilder's smoothing
    atr = np.zeros(len(high))
    plus_di = np.zeros(len(high))
    minus_di = np.zeros(len(high))
    
    atr[period-1] = np.mean(tr[:period])
    plus_dm_sum = np.sum(plus_dm[:period])
    minus_dm_sum = np.sum(minus_dm[:period])
    
    for i in range(period, len(high)):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        plus_dm_sum = plus_dm_sum - (plus_dm_sum/period) + plus_dm[i]
        minus_dm_sum = minus_dm_sum - (minus_dm_sum/period) + minus_dm[i]
        plus_di[i] = 100 * plus_dm_sum / atr[i] if atr[i] != 0 else 0
        minus_di[i] = 100 * minus_dm_sum / atr[i] if atr[i] != 0 else 0
    
    dx = np.zeros(len(high))
    adx = np.zeros(len(high))
    
    for i in range(period, len(high)):
        di_sum = plus_di[i] + minus_di[i]
        dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / di_sum if di_sum != 0 else 0
    
    adx[2*period-1] = np.sum(dx[period:2*period]) / period
    for i in range(2*period, len(high)):
        adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
    
    return adx

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ADX
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, period=14)
    adx_1d_4h = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate 4h VWAP (20-period)
    vwap_4h = calculate_vwap(high, low, close, volume, window=20)
    
    # Calculate volume ratio (current vs 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=1).mean().values
    volume_ratio = volume / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # need sufficient data for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if np.isnan(adx_1d_4h[i]) or np.isnan(vwap_4h[i]) or np.isnan(volume_ratio[i]):
            signals[i] = 0.0
            continue
        
        # ADX threshold: >25 indicates trending, <20 indicates ranging
        is_trending = adx_1d_4h[i] > 25
        is_ranging = adx_1d_4h[i] < 20
        
        # Price deviation from VWAP (%)
        vwap_deviation = (close[i] - vwap_4h[i]) / vwap_4h[i] * 100
        
        if position == 0:
            # In ranging market: mean reversion to VWAP
            if is_ranging and volume_ratio[i] > 1.5:  # volume confirmation
                if vwap_deviation < -1.0:  # price below VWAP -> long
                    signals[i] = 0.25
                    position = 1
                elif vwap_deviation > 1.0:  # price above VWAP -> short
                    signals[i] = -0.25
                    position = -1
            # In trending market: follow momentum
            elif is_trending and volume_ratio[i] > 1.3:
                if vwap_deviation > 0.5:  # price above VWAP and rising -> long
                    signals[i] = 0.25
                    position = 1
                elif vwap_deviation < -0.5:  # price below VWAP and falling -> short
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Long exit conditions
            if is_ranging and vwap_deviation > 0.2:  # reverted to VWAP in ranging
                signals[i] = -0.25  # reverse to short
                position = -1
            elif is_trending and vwap_deviation < -0.3:  # weak trend reversal
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25  # maintain long
        
        elif position == -1:
            # Short exit conditions
            if is_ranging and vwap_deviation < -0.2:  # reverted to VWAP in ranging
                signals[i] = 0.25  # reverse to long
                position = 1
            elif is_trending and vwap_deviation > 0.3:  # weak trend reversal
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25  # maintain short
    
    return signals

name = "4h_VWAP_ADX_MeanReversion_Trend"
timeframe = "4h"
leverage = 1.0