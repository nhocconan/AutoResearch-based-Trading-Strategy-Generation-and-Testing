#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla H3/L3 Breakout with 4h Trend Filter and Volume Confirmation.
- Uses Camarilla pivot levels (H3, L3) from prior 4h for high-probability breakout levels.
- Breakout above H3 or below L3 with volume confirmation captures short-term momentum.
- 4h EMA50 provides higher-timeframe trend filter to align with intermediate trend.
- Session filter (08-20 UTC) reduces noise during low-liquidity periods.
- Position size 0.20 balances profit and drawdown control while minimizing fee churn.
- Target trades: 60-150 total over 4 years (15-37/year) to stay within fee drag limits.
- Works in bull/bear markets via 4h trend filter and volatility-based logic.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data ONCE before loop for Camarilla pivots and EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # 4h EMA50 trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate Camarilla levels from prior 4h candle (H3, L3, H4, L4)
    # Camarilla: H4 = close + 1.1*(high-low)*1.1/2, H3 = close + 1.1*(high-low)*1.1/4
    #            L3 = close - 1.1*(high-low)*1.1/4, L4 = close - 1.1*(high-low)*1.1/2
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Camarilla levels for each 4h bar
    range_4h = high_4h - low_4h
    camarilla_h3 = close_4h + 1.1 * range_4h * 1.1 / 4
    camarilla_l3 = close_4h - 1.1 * range_4h * 1.1 / 4
    camarilla_h4 = close_4h + 1.1 * range_4h * 1.1 / 2
    camarilla_l4 = close_4h - 1.1 * range_4h * 1.1 / 2
    
    # Align Camarilla levels to 1h timeframe (wait for 4h bar to close)
    h3_4h_aligned = align_htf_to_ltf(prices, df_4h, camarilla_h3)
    l3_4h_aligned = align_htf_to_ltf(prices, df_4h, camarilla_l3)
    h4_4h_aligned = align_htf_to_ltf(prices, df_4h, camarilla_h4)
    l4_4h_aligned = align_htf_to_ltf(prices, df_4h, camarilla_l4)
    
    # Volume confirmation: > 1.5x 20-period average (balanced for 1h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50) + 1
    
    for i in range(start_idx, n):
        # Skip if data not ready or outside session
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(h3_4h_aligned[i]) or 
            np.isnan(l3_4h_aligned[i]) or np.isnan(vol_ma[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        volume_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Only trade with volume confirmation and in session
            if volume_confirm:
                # Long: break above H3 + above 4h EMA50 (bullish higher-timeframe trend)
                if close[i] > h3_4h_aligned[i] and close[i] > ema_50_4h_aligned[i]:
                    signals[i] = 0.20
                    position = 1
                # Short: break below L3 + below 4h EMA50 (bearish higher-timeframe trend)
                elif close[i] < l3_4h_aligned[i] and close[i] < ema_50_4h_aligned[i]:
                    signals[i] = -0.20
                    position = -1
        elif position == 1:
            # Long exit: price closes below L3 (reversal) OR below EMA50 (trend change)
            if close[i] < l3_4h_aligned[i] or close[i] < ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price closes above H3 (reversal) OR above EMA50 (trend change)
            if close[i] > h3_4h_aligned[i] or close[i] > ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_H3L3_Breakout_4hEMA50_VolumeConfirm_v1"
timeframe = "1h"
leverage = 1.0