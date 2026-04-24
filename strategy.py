#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla H3/L3 breakout with 4h EMA50 trend filter and volume spike confirmation.
- Long when price breaks above Camarilla H3 AND close > 4h EMA50 (bullish trend)
- Short when price breaks below Camarilla L3 AND close < 4h EMA50 (bearish trend)
- Volume must be > 2.0 * median volume of last 20 bars (volume spike filter to avoid fakeouts)
- Exit on opposite Camarilla breakout or trend reversal (close crosses 4h EMA50)
- Uses 1h primary timeframe with 4h HTF to target 60-150 total trades over 4 years (15-37/year)
- Camarilla H3/L3 provide strong intraday support/resistance levels
- 4h EMA50 ensures alignment with higher timeframe trend to reduce whipsaws in choppy markets
- Volume spike filter confirms institutional participation, reducing false breakouts
- Designed for BTC/ETH with edge in both trending and ranging markets
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 80:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla levels (based on previous bar's range)
    # H3 = close + 1.1*(high-low)/4, L3 = close - 1.1*(high-low)/4
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close[0] = np.nan
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    
    camarilla_h3 = prev_close + 1.1 * (prev_high - prev_low) / 4
    camarilla_l3 = prev_close - 1.1 * (prev_high - prev_low) / 4
    
    # Get 4h data ONCE before loop for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 4h EMA50 to 1h timeframe
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Volume spike filter: volume > 2.0 * median volume of last 20 bars
    vol_median = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    volume_spike = volume > (2.0 * vol_median)
    
    # Session filter: 08:00-20:00 UTC (reduce noise outside active trading hours)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50) + 1
    
    for i in range(start_idx, n):
        # Skip if data not ready or outside session
        if (np.isnan(camarilla_h3[i]) or np.isnan(camarilla_l3[i]) or 
            np.isnan(ema_50_4h_aligned[i]) or np.isnan(vol_median[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Camarilla H3, trend up (close > EMA50), volume spike, in session
            if close[i] > camarilla_h3[i] and close[i] > ema_50_4h_aligned[i] and volume_spike[i]:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below Camarilla L3, trend down (close < EMA50), volume spike, in session
            elif close[i] < camarilla_l3[i] and close[i] < ema_50_4h_aligned[i] and volume_spike[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: price breaks below Camarilla L3 OR trend reversal (close < EMA50)
            if close[i] < camarilla_l3[i] or close[i] < ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price breaks above Camarilla H3 OR trend reversal (close > EMA50)
            if close[i] > camarilla_h3[i] or close[i] > ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_H3L3_4hEMA50_VolumeSpike_Session_v1"
timeframe = "1h"
leverage = 1.0