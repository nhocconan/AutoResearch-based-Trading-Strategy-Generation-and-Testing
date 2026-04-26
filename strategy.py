#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeSpike_Session
Hypothesis: Camarilla R1/S1 breakout on 1h with 4h EMA20 trend filter and volume spike (>1.5x average volume). Uses 08-20 UTC session filter to avoid off-hours noise. Discrete position sizing (0.20) to minimize fee churn. Targets 15-30 trades/year by requiring confluence of 4h trend, Camarilla breakout, volume spike, and session alignment. Works in bull/bear by following 4h trend direction.
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
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # 4h EMA20 for trend filter
    ema_20_4h = pd.Series(df_4h['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # Load 1d data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla R1 and S1 levels (core breakout levels)
    camarilla_r1 = close_1d + (high_1d - low_1d) * 1.1 / 12
    camarilla_s1 = close_1d - (high_1d - low_1d) * 1.1 / 12
    
    # Align Camarilla levels to 1h timeframe (1 bar delay for completed 1d bar)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Calculate average volume for confirmation (24-period SMA = 6h average)
    avg_volume = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    # Session filter: 08-20 UTC (avoid Asian session noise and low liquidity)
    hours = prices.index.hour  # prices.index is DatetimeIndex
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.20
    
    # Start after warmup (need 50 for EMA, 24 for volume)
    start_idx = max(50, 24)
    
    for i in range(start_idx, n):
        # Get current values
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        ema_val = ema_20_4h_aligned[i]
        r1_val = camarilla_r1_aligned[i]
        s1_val = camarilla_s1_aligned[i]
        hour = hours[i]
        
        # Skip if any data not ready
        if (np.isnan(ema_val) or np.isnan(avg_vol) or np.isnan(r1_val) or np.isnan(s1_val)):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Session filter: only trade 08-20 UTC
        in_session = (8 <= hour <= 20)
        
        # Volume confirmation: current volume > 1.5x average volume (moderate breakout)
        volume_confirmed = vol > 1.5 * avg_vol
        
        # Long logic: price breaks above Camarilla R1 with 4h uptrend and volume confirmation
        long_condition = in_session and (close_val > r1_val) and (close_val > ema_val) and volume_confirmed
        # Short logic: price breaks below Camarilla S1 with 4h downtrend and volume confirmation
        short_condition = in_session and (close_val < s1_val) and (close_val < ema_val) and volume_confirmed
        
        # Exit logic: trend reversal (price crosses 4h EMA20) OR session end
        exit_long = (close_val < ema_val) or not in_session
        exit_short = (close_val > ema_val) or not in_session
        
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

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeSpike_Session"
timeframe = "1h"
leverage = 1.0