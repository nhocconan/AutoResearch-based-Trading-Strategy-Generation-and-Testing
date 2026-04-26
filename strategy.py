#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeConfirm
Hypothesis: Camarilla R1/S1 breakout on 1h with 4h EMA50 trend filter and volume confirmation (>1.8x average volume).
In bull markets: price breaks above R1 with 4h uptrend and high volume → long.
In bear markets: price breaks below S1 with 4h downtrend and high volume → short.
Uses discrete position sizing (0.20) to minimize fee churn. Target: 60-150 total trades over 4 years (15-37/year) on 1h timeframe.
Requires BTC/ETH edge via 4h trend and volume filters; avoids SOL-only bias by requiring trend alignment.
Session filter: 08-20 UTC to reduce noise trades.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:  # Need warmup for EMA and volume
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 4h data for HTF trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 5:
        return np.zeros(n)
    
    # 4h EMA50 for trend filter
    ema_50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate average volume for confirmation (20-period SMA)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC (precompute before loop)
    hours = prices.index.hour  # prices.index is DatetimeIndex
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.20
    
    # Start after warmup (need 20 for volume)
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if outside session
        if not in_session[i]:
            signals[i] = 0.0
            position = 0
            continue
        
        # Calculate Camarilla pivot points using previous day's OHLC
        # Need to get previous day's high, low, close
        current_time = prices.iloc[i]['open_time']
        # Get previous day's data (simplified: use 24h lookback for daily pivot)
        lookback_idx = max(0, i - 24)  # Approximate previous day (24 * 1h bars)
        if lookback_idx >= i:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
            
        prev_high = np.max(high[lookback_idx:i])
        prev_low = np.min(low[lookback_idx:i])
        prev_close = close[i-1]  # Previous bar close
        
        # Calculate pivot point
        pivot = (prev_high + prev_low + prev_close) / 3.0
        range_val = prev_high - prev_low
        
        # Camarilla levels
        R1 = pivot + (range_val * 1.1 / 12)
        S1 = pivot - (range_val * 1.1 / 12)
        
        close_val = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        ema_val = ema_50_4h_aligned[i]
        
        # Skip if any data not ready
        if np.isnan(R1) or np.isnan(S1) or np.isnan(ema_val) or np.isnan(avg_vol):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Volume confirmation: current volume > 1.8x average volume
        volume_confirmed = vol > 1.8 * avg_vol
        
        # Long logic: price breaks above R1 with 4h uptrend and volume confirmation
        long_condition = (close_val > R1) and (close_val > ema_val) and volume_confirmed
        # Short logic: price breaks below S1 with 4h downtrend and volume confirmation
        short_condition = (close_val < S1) and (close_val < ema_val) and volume_confirmed
        
        # Exit logic: trend reversal or opposite breakout
        exit_long = (close_val < ema_val) or (close_val < S1)
        exit_short = (close_val > ema_val) or (close_val > R1)
        
        if long_condition and position != 1:
            signals[i] = base_size
            position = 1
        elif short_condition and position != -1:
            signals[i] = -base_size
            position = -1
        elif position == 1 and exit_long:
            signals[i] = 0.0
            position = 0
        elif position == -1 and exit_short:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeConfirm"
timeframe = "1h"
leverage = 1.0