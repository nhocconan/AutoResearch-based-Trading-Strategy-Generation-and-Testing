#!/usr/bin/env python3
"""
1h_HighLow_Breakout_4hTrend_1dVolFilter_v1
Hypothesis: Trade 1h breakouts of prior session high/low with 4h EMA50 trend filter and 1d volume confirmation (1.5x median). Uses ATR(14) trailing stop (2.0x) and session filter (08-20 UTC). Designed for moderate trade frequency (~20-40/year) by requiring HTF trend alignment and volume confirmation. Works in bull markets (breakouts with uptrend) and bear markets (breakdowns with downtrend). Focus on BTC/ETH as primary targets.
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
    
    # Get 4h data for HTF trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Get 1d data for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 4h EMA(50) for trend filter
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # 1d median volume for volume filter
    vol_median_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).median().values
    
    # Align HTF indicators to 1h timeframe
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    vol_median_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_median_1d)
    
    # ATR(14) for stop (calculated on 1h)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour  # prices.index is DatetimeIndex
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    long_stop = 0.0
    short_stop = 0.0
    
    # Warmup: max of 4h EMA (50), 1d volume median (20), 1h ATR (14)
    start_idx = max(50, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(vol_median_1d_aligned[i]) or 
            np.isnan(atr_14[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        ema_50_4h_val = ema_50_4h_aligned[i]
        vol_median_1d_val = vol_median_1d_aligned[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        volume_val = volume[i]
        atr_14_val = atr_14[i]
        hour = hours[i]
        in_session = (8 <= hour <= 20)  # UTC 08-20
        
        if position == 0 and in_session:
            # Calculate prior session high/low (using prior 4h bar for session context)
            # For simplicity, use prior 20 bars (approx 1 session) high/low
            lookback = 20
            if i >= lookback:
                session_high = np.max(high[i-lookback:i])
                session_low = np.min(low[i-lookback:i])
                
                # Long: break above session high, uptrend (close > EMA50), volume > 1.5x median
                long_signal = (high_val > session_high) and \
                              (close_val > ema_50_4h_val) and \
                              (volume_val > 1.5 * vol_median_1d_val)
                # Short: break below session low, downtrend (close < EMA50), volume > 1.5x median
                short_signal = (low_val < session_low) and \
                               (close_val < ema_50_4h_val) and \
                               (volume_val > 1.5 * vol_median_1d_val)
                
                if long_signal:
                    signals[i] = 0.20
                    position = 1
                    entry_price = close_val
                    long_stop = entry_price - 2.0 * atr_14_val
                elif short_signal:
                    signals[i] = -0.20
                    position = -1
                    entry_price = close_val
                    short_stop = entry_price + 2.0 * atr_14_val
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0
        elif position == 0:
            # Outside session: stay flat
            signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.20
            # Update trailing stop: move stop up as price makes new highs
            long_stop = max(long_stop, high_val - 2.0 * atr_14_val)
            # Exit: trailing stop hit or trend reversal (close < EMA50)
            if (low_val < long_stop) or (close_val < ema_50_4h_val):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.20
            # Update trailing stop: move stop down as price makes new lows
            short_stop = min(short_stop, low_val + 2.0 * atr_14_val)
            # Exit: trailing stop hit or trend reversal (close > EMA50)
            if (high_val > short_stop) or (close_val > ema_50_4h_val):
                signals[i] = 0.0
                position = 0
        else:
            # Outside session with position: hold until exit conditions
            if position == 1:
                signals[i] = 0.20
                long_stop = max(long_stop, high_val - 2.0 * atr_14_val)
                if (low_val < long_stop) or (close_val < ema_50_4h_val):
                    signals[i] = 0.0
                    position = 0
            elif position == -1:
                signals[i] = -0.20
                short_stop = min(short_stop, low_val + 2.0 * atr_14_val)
                if (high_val > short_stop) or (close_val > ema_50_4h_val):
                    signals[i] = 0.0
                    position = 0
    
    return signals

name = "1h_HighLow_Breakout_4hTrend_1dVolFilter_v1"
timeframe = "1h"
leverage = 1.0