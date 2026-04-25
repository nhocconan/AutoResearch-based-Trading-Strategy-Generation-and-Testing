#!/usr/bin/env python3
"""
1d_Camarilla_H1L1_Breakout_1wTrend_VolumeSpike_v1
Hypothesis: Trade Camarilla H1/L1 breakouts on daily timeframe with 1w trend filter and volume spike confirmation.
In bull markets (price > weekly EMA34): buy when price breaks above H1 level with volume > 1.5x average.
In bear markets (price < weekly EMA34): sell when price breaks below L1 level with volume > 1.5x average.
Exit on opposite Camarilla level touch (H4/L4) or trend reversal.
Position size: 0.25 to manage drawdown.
Target: 15-25 trades/year to stay well under 150-trade 1d hard max.
Uses proven Camarilla structure with volume confirmation for BTC/ETH edge.
Works in bull (breakouts with uptrend) and bear (breakdowns with downtrend) markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 35:  # Need at least 35 bars for EMA34
        return np.zeros(n)
    
    # Calculate 1w EMA34 for HTF trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate average volume for spike detection (20-period SMA)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for volume SMA (20)
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if HTF data not ready
        if np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_sma_20[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Calculate Camarilla levels from previous day's range
        if i >= 1:
            prev_high = high[i-1]
            prev_low = low[i-1]
            prev_close = close[i-1]
            daily_range = prev_high - prev_low
            
            # Camarilla levels
            h4 = prev_close + daily_range * 1.1 / 2
            h3 = prev_close + daily_range * 1.1 / 4
            h2 = prev_close + daily_range * 1.1 / 6
            h1 = prev_close + daily_range * 1.1 / 12
            l1 = prev_close - daily_range * 1.1 / 12
            l2 = prev_close - daily_range * 1.1 / 6
            l3 = prev_close - daily_range * 1.1 / 4
            l4 = prev_close - daily_range * 1.1 / 2
        else:
            # Not enough data for Camarilla calculation
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine 1w HTF trend (bullish = price above weekly EMA34)
        htf_1w_bullish = close[i] > ema_34_1w_aligned[i]
        htf_1w_bearish = close[i] < ema_34_1w_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_spike = volume[i] > 1.5 * vol_sma_20[i]
        
        if position == 0:
            # Long setup: price breaks above H1 + 1w uptrend + volume spike
            long_setup = (close[i] > h1) and htf_1w_bullish and volume_spike
            
            # Short setup: price breaks below L1 + 1w downtrend + volume spike
            short_setup = (close[i] < l1) and htf_1w_bearish and volume_spike
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: price touches L4 (stop) OR 1w trend turns bearish
            if (close[i] <= l4) or (not htf_1w_bullish):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price touches H4 (stop) OR 1w trend turns bullish
            if (close[i] >= h4) or (htf_1w_bullish):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Camarilla_H1L1_Breakout_1wTrend_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0