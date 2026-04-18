#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h volume-weighted VWAP mean reversion with 4h trend filter and 1d volatility filter.
# In ranging markets (1d ATR low), price reverts to VWAP with high probability.
# 4h trend filter ensures we only trade mean reversion in sideways markets, not strong trends.
# Volume confirmation filters low-conviction moves.
# Designed for low trade frequency (15-35/year) to minimize fee drag.
# Works in bull markets (mean reversion during pullbacks) and bear markets (mean reversion during bounces).

name = "1h_VWAP_MeanRev_4hTrend_1dVolFilter"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend filter (ONCE before loop)
    df_4h = get_htf_data(prices, '4h')
    # Get 1d data for volatility filter (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate VWAP (typical price * volume cumulative)
    typical_price = (high + low + close) / 3
    vwap_numerator = np.cumsum(typical_price * volume)
    vwap_denominator = np.cumsum(volume)
    vwap = np.divide(vwap_numerator, vwap_denominator, 
                     out=np.full_like(vwap_numerator, np.nan), 
                     where=vwap_denominator!=0)
    
    # Calculate 4h EMA34 for trend filter (using previous bar to avoid look-ahead)
    ema34_4h = pd.Series(df_4h['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_4h_shifted = np.roll(ema34_4h, 1)
    ema34_4h_shifted[0] = np.nan
    ema34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema34_4h_shifted)
    
    # Calculate 1d ATR for volatility filter (using Wilder's smoothing)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Wilder's smoothing (equivalent to EMA with alpha=1/period)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            result[period-1] = np.nansum(data[:period]) / period
            for i in range(period, len(data)):
                if not np.isnan(result[i-1]) and not np.isnan(data[i]):
                    result[i] = result[i-1] * (1 - 1/period) + data[i] * (1/period)
                else:
                    result[i] = np.nan
        return result
    
    atr_14 = wilders_smoothing(tr, 14)
    atr_14_shifted = np.roll(atr_14, 1)
    atr_14_shifted[0] = np.nan
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14_shifted, additional_delay_bars=0)
    
    # Calculate 1d ATR percentile (20-period lookback) for volatility regime
    atr_ratio = pd.Series(atr_14_aligned).rolling(window=20, min_periods=20).apply(
        lambda x: np.percentile(x, 50) if len(x) == 20 else np.nan, raw=True).values
    
    # Session filter: 08-20 UTC
    hour_index = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(vwap[i]) or np.isnan(ema34_4h_aligned[i]) or 
            np.isnan(atr_14_aligned[i]) or np.isnan(atr_ratio[i])):
            signals[i] = 0.0
            continue
        
        hour = hour_index[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            signals[i] = 0.0
            continue
        
        # Volatility filter: only trade when ATR is below median (low volatility ranging market)
        low_volatility = atr_14_aligned[i] < atr_ratio[i]
        
        # Trend filter: only trade when price is near 4h EMA (sideways relative to trend)
        near_trend = abs(close[i] - ema34_4h_aligned[i]) / ema34_4h_aligned[i] < 0.02
        
        if position == 0:
            # Long: price below VWAP in low volatility, sideways market
            if low_volatility and near_trend and close[i] < vwap[i]:
                signals[i] = 0.20
                position = 1
            # Short: price above VWAP in low volatility, sideways market
            elif low_volatility and near_trend and close[i] > vwap[i]:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: price crosses above VWAP or volatility increases
            if close[i] > vwap[i] or not low_volatility:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: price crosses below VWAP or volatility increases
            if close[i] < vwap[i] or not low_volatility:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals