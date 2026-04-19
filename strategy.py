#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h momentum breakout with 4h trend filter and 1d volume confirmation
# - 1h EMA(21) for momentum direction: long when price > EMA21, short when price < EMA21
# - 4h EMA(50) trend filter: only take longs when price > 4h EMA50, shorts when price < 4h EMA50
# - 1d volume > 1.5x 20-period average for conviction
# - Session filter: only trade 08-20 UTC to avoid low liquidity periods
# - Fixed position size of 0.20 to limit risk
# - Designed to work in both bull and bear markets by following higher timeframe trend
# - Target: 15-30 trades/year to avoid excessive fee drag

name = "1h_EMA21_4hTrend_1dVolume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    
    # 4h EMA(50) for trend direction
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    
    # 1d volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # 1h EMA(21) for momentum
    ema_21_1h = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(ema_50_4h_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]) or np.isnan(ema_21_1h[i]):
            signals[i] = 0.0
            continue
            
        # Check session: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
            
        # Volume filter: current 1h volume > 1.5x 1d average volume (scaled)
        # Scale 1d average to 1h: 1d has 24x 1h bars, so divide by 24
        volume_filter = vol_ma_1d_aligned[i] > 0 and volume[i] > 1.5 * (vol_ma_1d_aligned[i] / 24.0)
        
        if position == 0:
            # Look for long entry: uptrend (price > 4h EMA50) + bullish momentum (price > 1h EMA21) + volume
            if close[i] > ema_50_4h_aligned[i] and close[i] > ema_21_1h[i] and volume_filter:
                signals[i] = 0.20
                position = 1
            # Look for short entry: downtrend (price < 4h EMA50) + bearish momentum (price < 1h EMA21) + volume
            elif close[i] < ema_50_4h_aligned[i] and close[i] < ema_21_1h[i] and volume_filter:
                signals[i] = -0.20
                position = -1
                
        elif position == 1:
            # Long position: exit when momentum turns bearish or trend breaks
            if close[i] < ema_21_1h[i] or close[i] < ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
                
        elif position == -1:
            # Short position: exit when momentum turns bullish or trend breaks
            if close[i] > ema_21_1h[i] or close[i] > ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals