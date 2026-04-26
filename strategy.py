#!/usr/bin/env python3
"""
1d_Camarilla_Pivot_Reversal_1wTrend_VolumeConfirm
Hypothesis: Daily mean reversion at Camarilla H3/L3 levels with 1-week trend filter.
- Long when price crosses below Camarilla L3 AND 1w EMA50 uptrend AND volume > 1.8 * volume_ma(20)
- Short when price crosses above Camarilla H3 AND 1w EMA50 downtrend AND volume > 1.8 * volume_ma(20)
- Uses Camarilla pivot levels from daily chart for structure-based reversals
- 1-week EMA50 filter ensures trading with higher timeframe trend to avoid counter-trend whipsaws
- Volume spike (1.8x) confirms institutional participation at extreme levels
- Exit on opposite Camarilla level (H3 for longs, L3 for shorts) or trend reversal
- Designed for low frequency (target 10-25 trades/year on 1d) to minimize fee drag
- Novelty: Camarilla H3/L3 reversal with weekly trend filter on daily timeframe
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:  # Need enough data for calculations
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop for trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w EMA50 for trend filter (needs completed 1w candle)
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    # Trend: 1 = uptrend (close > EMA50), -1 = downtrend (close < EMA50), 0 = neutral/invalid
    trend_1w = np.where(ema_50_1w_aligned > 0, 
                        np.where(close > ema_50_1w_aligned, 1, -1), 
                        0)
    
    # Calculate Camarilla pivot levels on daily chart (primary timeframe)
    # Using previous day's OHLC for Camarilla calculation
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close[0] = close[0]  # first bar uses current values
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    
    # Camarilla calculations
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_hl = prev_high - prev_low
    
    # Resistance levels
    H3 = pivot + (range_hl * 1.1 / 4.0)
    H4 = pivot + (range_hl * 1.1 / 2.0)
    # Support levels
    L3 = pivot - (range_hl * 1.1 / 4.0)
    L4 = pivot - (range_hl * 1.1 / 2.0)
    
    # Calculate volume filter: volume > 1.8 * volume_ma(20) for confirmation
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 50 for 1w EMA, 20 for volume MA)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(H3[i]) or np.isnan(L3[i]) or
            np.isnan(trend_1w[i]) or np.isnan(volume_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Camarilla H3/L3 reversal conditions with trend and volume spike filter
        if position == 0:
            # Long: Price crosses below Camarilla L3 AND 1w uptrend AND volume spike
            if close[i] < L3[i] and close[i-1] >= L3[i-1] and trend_1w[i] == 1 and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price crosses above Camarilla H3 AND 1w downtrend AND volume spike
            elif close[i] > H3[i] and close[i-1] <= H3[i-1] and trend_1w[i] == -1 and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: Price rises above Camarilla H3 OR 1w trend turns down
            if close[i] > H3[i] or trend_1w[i] == -1:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: Price falls below Camarilla L3 OR 1w trend turns up
            if close[i] < L3[i] or trend_1w[i] == 1:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Camarilla_Pivot_Reversal_1wTrend_VolumeConfirm"
timeframe = "1d"
leverage = 1.0