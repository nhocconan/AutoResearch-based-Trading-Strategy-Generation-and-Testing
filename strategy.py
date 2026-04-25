#!/usr/bin/env python3
"""
1h Camarilla Pivot H3L3 Breakout with 4h EMA34 Trend and Volume Spike Filter
Hypothesis: Camarilla H3/L3 levels act as intraday support/resistance. Breakouts above H3 or below L3 with volume confirmation and 4h EMA34 trend alignment capture momentum moves. Works in bull markets (long H3 breakouts) and bear markets (short L3 breakdowns) by requiring 4h trend alignment. Uses 1h only for entry timing precision, targeting 15-35 trades/year.
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
    
    # Get 4h data for EMA34 trend filter (call ONCE before loop)
    df_4h = get_htf_data(prices, '4h')
    
    if len(df_4h) < 34:
        return np.zeros(n)
    
    # Calculate 4h EMA34 for trend filter
    ema_34_4h = pd.Series(df_4h['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Calculate daily Camarilla pivots from prior day's OHLC
    # We need to resample to 1d to get prior day's OHLC, but use actual Binance 1d data
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Prior day's OHLC (yesterday's close is the most recent completed 1d bar)
    prev_close = df_1d['close'].iloc[-2] if len(df_1d) >= 2 else df_1d['close'].iloc[-1]
    prev_high = df_1d['high'].iloc[-2] if len(df_1d) >= 2 else df_1d['high'].iloc[-1]
    prev_low = df_1d['low'].iloc[-2] if len(df_1d) >= 2 else df_1d['low'].iloc[-1]
    
    # Camarilla pivot calculations
    range_ = prev_high - prev_low
    H3 = prev_close + range_ * 1.1 / 4
    L3 = prev_close - range_ * 1.1 / 4
    H4 = prev_close + range_ * 1.1 / 2
    L4 = prev_close - range_ * 1.1 / 2
    
    # Align Camarilla levels to 1h timeframe (same value for all bars until new 1d bar)
    H3_arr = np.full(n, H3)
    L3_arr = np.full(n, L3)
    H4_arr = np.full(n, H4)
    L4_arr = np.full(n, L4)
    
    # Volume spike filter: volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for volume MA and valid data
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if np.isnan(ema_34_4h_aligned[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_volume_spike = volume_spike[i]
        ema_trend = ema_34_4h_aligned[i]
        H3_level = H3_arr[i]
        L3_level = L3_arr[i]
        
        if position == 0:
            # Look for entry signals
            # Long: price breaks above H3 with volume spike AND price > 4h EMA34 (uptrend)
            long_entry = (curr_close > H3_level) and curr_volume_spike and (curr_close > ema_trend)
            # Short: price breaks below L3 with volume spike AND price < 4h EMA34 (downtrend)
            short_entry = (curr_close < L3_level) and curr_volume_spike and (curr_close < ema_trend)
            
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
            # Exit: price falls back below H3 (failed breakout) OR price < 4h EMA34 (trend change)
            if (curr_close < H3_level) or (curr_close < ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short position management
            # Exit: price rises back above L3 (failed breakdown) OR price > 4h EMA34 (trend change)
            if (curr_close > L3_level) or (curr_close > ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_H3L3_Breakout_4hEMA34_Trend_VolumeSpike"
timeframe = "1h"
leverage = 1.0