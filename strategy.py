#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolumeSpike_v1
Hypothesis: Trade Camarilla pivot breakouts on 1h with 4h EMA34 trend filter and 1d volume spike confirmation.
Long when price breaks above R1 and 4h EMA34 uptrend + 1d volume > 1.5x 20-period average.
Short when price breaks below S1 and 4h EMA34 downtrend + 1d volume > 1.5x 20-period average.
Exit on opposite Camarilla level touch or trend reversal.
Session filter: 08-20 UTC to avoid low-liquidity hours.
Position size: 0.20 to limit drawdown and enable multiple concurrent positions.
Target: 15-35 trades/year (~60-140 over 4 years) to stay under 200 trade hard max.
Works in bull (breakouts with uptrend) and bear (breakdowns with downtrend) markets.
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
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for HTF trend filter (EMA34)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 35:  # Need warmup for EMA34
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Get 1d data for volume spike filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:  # Need warmup for volume average
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Calculate Camarilla pivots from previous 1d OHLC
    # Camarilla: R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    camarilla_multiplier = 1.1 / 12
    R1 = close_1d + camarilla_multiplier * (high_1d - low_1d)
    S1 = close_1d - camarilla_multiplier * (high_1d - low_1d)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for the longest indicator (EMA34_4h: 34 bars)
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        # Skip if data not ready
        if (np.isnan(ema_34_4h_aligned[i]) or 
            np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or
            np.isnan(vol_ma_20_1d_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        # Determine 4h HTF trend (bullish = price above EMA34)
        htf_4h_bullish = close[i] > ema_34_4h_aligned[i]
        htf_4h_bearish = close[i] < ema_34_4h_aligned[i]
        
        # Determine 1d volume spike (current volume > 1.5x 20-period average)
        volume_spike = volume[i] > 1.5 * vol_ma_20_1d_aligned[i]
        
        if position == 0:
            # Long setup: price breaks above R1 + 4h uptrend + volume spike
            long_setup = (close[i] > R1_aligned[i]) and htf_4h_bullish and volume_spike
            
            # Short setup: price breaks below S1 + 4h downtrend + volume spike
            short_setup = (close[i] < S1_aligned[i]) and htf_4h_bearish and volume_spike
            
            if long_setup:
                signals[i] = 0.20
                position = 1
            elif short_setup:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.20
            # Exit: price touches S1 (stop) OR 4h trend turns bearish
            if (close[i] <= S1_aligned[i]) or (not htf_4h_bullish):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.20
            # Exit: price touches R1 (stop) OR 4h trend turns bullish
            if (close[i] >= R1_aligned[i]) or (htf_4h_bullish):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolumeSpike_v1"
timeframe = "1h"
leverage = 1.0