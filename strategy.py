#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using 1w Commodity Channel Index (CCI) for overbought/oversold
# and 1w ADX for trend strength. Enters long when CCI crosses above -100 (oversold recovery) 
# in weak trend (ADX < 25) with volume confirmation. Enters short when CCI crosses below 100 
# (overbought correction) in weak trend with volume confirmation. Exits when CCI crosses 
# zero (mean reversion) or trend strengthens (ADX > 25). Designed to capture mean reversion
# in ranging markets while avoiding strong trends. Target: 30-100 total trades over 4 years.

name = "1d_1wCCI_ADX_MeanReversion"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for CCI and ADX calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w CCI (20 period)
    cci_period = 20
    typical_price = (high_1w + low_1w + close_1w) / 3
    sma_tp = pd.Series(typical_price).rolling(window=cci_period, min_periods=cci_period).mean().values
    mad = pd.Series(typical_price).rolling(window=cci_period, min_periods=cci_period).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=False
    ).values
    # Avoid division by zero
    mad = np.where(mad == 0, 1e-10, mad)
    cci = (typical_price - sma_tp) / (0.015 * mad)
    
    # Calculate 1w ADX (14 period)
    adx_period = 14
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    dm_plus = np.where((high_1w - np.roll(high_1w, 1)) > (np.roll(low_1w, 1) - low_1w),
                       np.maximum(high_1w - np.roll(high_1w, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1w, 1) - low_1w) > (high_1w - np.roll(high_1w, 1)),
                        np.maximum(np.roll(low_1w, 1) - low_1w, 0), 0)
    
    # Wilder's smoothing for TR, DM+, DM-
    def wilders_smoothing(data, period):
        result = np.zeros_like(data)
        if len(data) < period:
            return result
        result[period-1] = np.nansum(data[:period])
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    tr_smooth = wilders_smoothing(tr, adx_period)
    dm_plus_smooth = wilders_smoothing(dm_plus, adx_period)
    dm_minus_smooth = wilders_smoothing(dm_minus, adx_period)
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / tr_smooth
    di_minus = 100 * dm_minus_smooth / tr_smooth
    # Avoid division by zero
    dx_denom = di_plus + di_minus
    dx_denom = np.where(dx_denom == 0, 1e-10, dx_denom)
    dx = 100 * np.abs(di_plus - di_minus) / dx_denom
    # ADX is smoothed DX
    adx = wilders_smoothing(dx, adx_period)
    
    # Align 1w indicators to daily timeframe
    cci_aligned = align_htf_to_ltf(prices, df_1w, cci)
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # Volume filter (daily timeframe)
    vol_ma_10 = pd.Series(volume).rolling(window=10, min_periods=10).mean().values
    volume_spike = volume > (1.5 * vol_ma_10)  # Moderate volume confirmation
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup
        # Skip if any critical value is NaN
        if (np.isnan(cci_aligned[i]) or np.isnan(adx_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Look for mean reentry opportunities in weak trend
            weak_trend = adx_aligned[i] < 25
            if weak_trend:
                # Long: CCI crosses above -100 from oversold with volume
                if cci_aligned[i] > -100 and cci_aligned[i-1] <= -100 and volume_spike[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: CCI crosses below 100 from overbought with volume
                elif cci_aligned[i] < 100 and cci_aligned[i-1] >= 100 and volume_spike[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: CCI crosses above zero (overbought) or trend strengthens
            if cci_aligned[i] > 0 and cci_aligned[i-1] <= 0 or adx_aligned[i] > 25:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: CCI crosses below zero (oversold) or trend strengthens
            if cci_aligned[i] < 0 and cci_aligned[i-1] >= 0 or adx_aligned[i] > 25:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals