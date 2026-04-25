#!/usr/bin/env python3
"""
1h Camarilla H3L3 Breakout + Volume Spike + 4h EMA50 Trend Filter
Hypothesis: Camarilla pivot levels on 1d provide robust support/resistance. 
Breakouts above H3 or below L3 with volume confirmation indicate institutional participation.
4h EMA50 filter ensures trades align with intermediate trend, reducing false breakouts.
Session filter (08-20 UTC) avoids low-liquidity periods. Discrete sizing (0.0, ±0.20) 
minimizes fee churn. Target: 15-30 trades/year on 1h.
Uses 1d for pivots and 4h for trend (MTF alignment). Works in bull markets via breakouts 
with trend and in bear markets via trend filter (avoids counter-trend entries).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for pivot calculation (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Get 4h data for EMA trend filter (call ONCE before loop)
    df_4h = get_htf_data(prices, '4h')
    
    if len(df_1d) < 20 or len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 1d Camarilla pivots from previous 1d OHLC
    prev_close = df_1d['close'].values
    prev_high = df_1d['high'].values
    prev_low = df_1d['low'].values
    rang = prev_high - prev_low
    H3 = prev_close + 1.0 * rang
    L3 = prev_close - 1.0 * rang
    
    # Align Camarilla levels to 1h (use previous day's levels for current day's trading)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    
    # Calculate 4h EMA50 for trend filter
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate volume spike: current volume > 2.0 * 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Session filter: 08-20 UTC (avoid low-liquidity periods)
    hours = prices.index.hour  # prices.index is DatetimeIndex, .hour works directly
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for volume MA and EMA
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or
            np.isnan(ema_50_4h_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        H3_level = H3_aligned[i]
        L3_level = L3_aligned[i]
        ema_trend = ema_50_4h_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Look for entry signals
            # Long: price breaks above H3 AND volume spike AND price > 4h EMA50 (uptrend)
            long_entry = (curr_close > H3_level) and vol_spike and (curr_close > ema_trend)
            # Short: price breaks below L3 AND volume spike AND price < 4h EMA50 (downtrend)
            short_entry = (curr_close < L3_level) and vol_spike and (curr_close < ema_trend)
            
            if long_entry:
                signals[i] = 0.20
                position = 1
            elif short_entry:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: price crosses below L3 (reversal) OR price < 4h EMA50 (trend change)
            if (curr_close < L3_level) or (curr_close < ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short position management
            # Exit: price crosses above H3 (reversal) OR price > 4h EMA50 (trend change)
            if (curr_close > H3_level) or (curr_close > ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_H3L3_Breakout_VolumeSpike_4hEMA50_Trend_Session"
timeframe = "1h"
leverage = 1.0