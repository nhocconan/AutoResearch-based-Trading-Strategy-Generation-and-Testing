#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Trend Composite (EMA20 > EMA50) and 1d Range Filter (price within 1d ATR bands) for direction.
# Entry on 1h pullback to EMA20 with volume confirmation (>1.5x average).
# Designed for low trade frequency (15-35/year) to minimize fee drag.
# Works in bull/bear: 4h trend filter avoids counter-trend, 1d range filter avoids extreme volatility.
# Position size fixed at 0.20 for consistent risk.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 4h data ONCE for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate EMA20 and EMA50 on 4h close
    close_4h = df_4h['close'].values
    ema20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Trend: 1 if EMA20 > EMA50, -1 otherwise
    trend_4h = np.where(ema20_4h > ema50_4h, 1, -1)
    
    # Align 4h trend to 1h
    trend_4h_aligned = align_htf_to_ltf(prices, df_4h, trend_4h.astype(float))
    
    # Load 1d data ONCE for range filter (ATR-based)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate 14-day ATR
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    atr_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # 1d range: ±1.0 * ATR from close
    upper_1d = close_1d + 1.0 * atr_1d
    lower_1d = close_1d - 1.0 * atr_1d
    
    # Align 1d range to 1h
    upper_1d_aligned = align_htf_to_ltf(prices, df_1d, upper_1d)
    lower_1d_aligned = align_htf_to_ltf(prices, df_1d, lower_1d)
    
    # 1h EMA20 for entry
    ema20_1h = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Volume confirmation: 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.20  # Fixed 20% position
    
    # Start after enough data for calculations
    start = max(20, 20, 14)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(trend_4h_aligned[i]) or
            np.isnan(upper_1d_aligned[i]) or
            np.isnan(lower_1d_aligned[i]) or
            np.isnan(ema20_1h[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: 08-20 UTC
        hour = pd.Timestamp(prices['open_time'].iloc[i]).hour
        in_session = 8 <= hour <= 20
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: 4h uptrend, price near 1d lower bound, pullback to 1h EMA20
            if (trend_4h_aligned[i] == 1 and
                low[i] <= lower_1d_aligned[i] * 1.02 and  # Allow 2% buffer
                abs(close[i] - ema20_1h[i]) / ema20_1h[i] < 0.015 and  # Within 1.5% of EMA20
                volume_confirmed and
                in_session):
                position = 1
                signals[i] = position_size
            # Short: 4h downtrend, price near 1d upper bound, pullback to 1h EMA20
            elif (trend_4h_aligned[i] == -1 and
                  high[i] >= upper_1d_aligned[i] * 0.98 and  # Allow 2% buffer
                  abs(close[i] - ema20_1h[i]) / ema20_1h[i] < 0.015 and  # Within 1.5% of EMA20
                  volume_confirmed and
                  in_session):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: 4h trend turns down OR price reaches 1d upper bound
            if (trend_4h_aligned[i] == -1 or
                high[i] >= upper_1d_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: 4h trend turns up OR price reaches 1d lower bound
            if (trend_4h_aligned[i] == 1 or
                low[i] <= lower_1d_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1h_4hTrend_1dRange_EMA20Pullback_VolumeFilter_v1"
timeframe = "1h"
leverage = 1.0