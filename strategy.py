#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h breakout strategy using 1d Donchian channels with volume confirmation and 1w volatility regime filter
# - Uses 1d Donchian breakout (20-period) for entry signals
# - Uses 1w ATR percentile to identify low volatility regimes (squeeze) - enters only when volatility is low
# - Requires volume spike (>2x 20-period average) for confirmation
# - Exits when price returns to the 1d midpoint or volatility expands
# - Designed to capture breakouts from low volatility periods with institutional volume
# - Target: 80-160 total trades over 4 years (20-40/year) with 0.25 position sizing

name = "4h_1dDonchian_1wVolSqueeze_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Get 1w data for volatility regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate 1d Donchian channels (20-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Upper band: highest high of last 20 periods
    high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    # Lower band: lowest low of last 20 periods
    low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    # Middle band: average of upper and lower
    donchian_mid = (high_20 + low_20) / 2
    
    # Calculate 1w ATR for volatility regime (14-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # ATR using Wilder's smoothing (equivalent to RMA)
    def wilders_smoothing(data, period):
        result = np.zeros_like(data)
        if len(data) < period:
            return result
        result[period-1] = np.nansum(data[:period])
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    atr_14 = wilders_smoothing(tr, 14)
    
    # Calculate ATR percentile rank (lookback 50 periods)
    atr_series = pd.Series(atr_14)
    atr_percentile = atr_series.rolling(window=50, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    
    # Align 1d indicators to 4h timeframe
    high_20_4h = align_htf_to_ltf(prices, df_1d, high_20)
    low_20_4h = align_htf_to_ltf(prices, df_1d, low_20)
    donchian_mid_4h = align_htf_to_ltf(prices, df_1d, donchian_mid)
    
    # Align 1w ATR percentile to 4h timeframe
    atr_percentile_4h = align_htf_to_ltf(prices, df_1w, atr_percentile)
    
    # Volume filters (4h timeframe)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)  # Strong volume confirmation
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup
        # Skip if any critical value is NaN
        if (np.isnan(high_20_4h[i]) or np.isnan(low_20_4h[i]) or np.isnan(donchian_mid_4h[i]) or
            np.isnan(atr_percentile_4h[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Look for low volatility regime (ATR < 30th percentile) 
            low_vol_regime = atr_percentile_4h[i] < 30
            
            if low_vol_regime:
                # Long: price breaks above 1d Donchian upper with volume spike
                if close[i] > high_20_4h[i] and volume_spike[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: price breaks below 1d Donchian lower with volume spike
                elif close[i] < low_20_4h[i] and volume_spike[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: price returns to midpoint OR volatility expands (ATR > 70th percentile)
            if close[i] < donchian_mid_4h[i] or atr_percentile_4h[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to midpoint OR volatility expands (ATR > 70th percentile)
            if close[i] > donchian_mid_4h[i] or atr_percentile_4h[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals