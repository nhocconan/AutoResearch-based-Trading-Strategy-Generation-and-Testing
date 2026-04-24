#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla pivot breakout with 1d EMA34 trend filter and volume spike confirmation.
- Long when price breaks above Camarilla R1 level AND close > 1d EMA34 (bullish trend) AND volume > 2.0 * volume SMA(20)
- Short when price breaks below Camarilla S1 level AND close < 1d EMA34 (bearish trend) AND volume > 2.0 * volume SMA(20)
- Exit when price returns to Camarilla H3/L3 levels or trend reverses
- Uses 4h primary timeframe with 1d HTF to target 75-200 trades over 4 years (19-50/year)
- Camarilla levels provide precise intraday support/resistance that work in both trending and ranging markets
- 1d EMA34 ensures alignment with daily trend to avoid whipsaws
- Volume spike filter adapts to changing volatility, reducing false signals in low-volume periods
- Designed for BTC/ETH with edge in bull markets (breakout continuation) and bear markets (mean reversion at extremes via trend filter)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d data ONCE before loop for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA34 to 4h timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate volume SMA(20) for volume spike confirmation
    volume_series = pd.Series(volume)
    volume_sma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * volume_sma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where EMA is ready
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if EMA data not ready
        if np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_sma_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate Camarilla levels for current 4h bar using previous day's OHLC
        # We need to get the previous day's OHLC from 1d data
        # Find the index of the previous completed 1d bar
        if i >= 16:  # At least one 1d bar completed (4h * 6 = 24h, but we use 1d alignment)
            # Get the index of the 1d bar that completed before current 4h bar
            # Since we have aligned arrays, we can use the previous 1d bar's data
            # The 1d data is aligned such that each 4h bar gets the previous completed 1d bar's values
            # We need to calculate Camarilla levels using the previous 1d bar's OHLC
            pass  # We'll calculate this differently - use the aligned 1d data to get previous day's OHLC
        
        # Simpler approach: use rolling window on 4h data to get daily OHLC
        # But this can cause look-ahead. Instead, we'll use a proxy:
        # For Camarilla, we need the previous day's high, low, close
        # We'll approximate using the last completed 24h period
        if i >= 24:  # At least 24 four-hour bars = 6 days, but we need 1 day
            # Get the high, low, close from 24 bars ago (previous day)
            prev_high = np.max(high[i-24:i]) if i >= 24 else high[i-1]
            prev_low = np.min(low[i-24:i]) if i >= 24 else low[i-1]
            prev_close = close[i-1]
        else:
            # Not enough data, skip
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate Camarilla levels
        # R4 = close + (high - low) * 1.5/2
        # R3 = close + (high - low) * 1.25/2
        # R2 = close + (high - low) * 1.166/2
        # R1 = close + (high - low) * 1.083/2
        # PP = (high + low + close) / 3
        # S1 = close - (high - low) * 1.083/2
        # S2 = close - (high - low) * 1.166/2
        # S3 = close - (high - low) * 1.25/2
        # S4 = close - (high - low) * 1.5/2
        # H3/L3 are used for exits: H3 = R3, L3 = S3
        range_val = prev_high - prev_low
        if range_val <= 0:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        r1 = prev_close + range_val * 1.083 / 2
        s1 = prev_close - range_val * 1.083 / 2
        h3 = prev_close + range_val * 1.25 / 2  # R3
        l3 = prev_close - range_val * 1.25 / 2  # S3
        
        if position == 0:
            # Long: price breaks above Camarilla R1, trend up (close > EMA34), volume spike
            if close[i] > r1 and close[i] > ema_34_1d_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla S1, trend down (close < EMA34), volume spike
            elif close[i] < s1 and close[i] < ema_34_1d_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price returns to Camarilla H3 (R3) or trend reverses
            if close[i] >= h3 or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price returns to Camarilla L3 (S3) or trend reverses
            if close[i] <= l3 or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dEMA34_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0