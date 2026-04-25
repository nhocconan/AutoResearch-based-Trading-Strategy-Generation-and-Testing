#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_12hEMA34_Trend_VolumeConfirm_v1
Hypothesis: Trade Donchian(20) breakouts on 4h with 12h EMA34 trend filter and volume spike confirmation.
In bull markets: buy when price breaks above upper Donchian channel + price > 12h EMA34 + volume > 1.5x MA20.
In bear markets: sell when price breaks below lower Donchian channel + price < 12h EMA34 + volume > 1.5x MA20.
Exit on opposite Donchian breakout or trend reversal.
Position size: 0.25 to limit drawdown and reduce fee churn.
Target: 20-50 trades/year to stay well under 400-trade 4h hard max.
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
    
    # Get 12h data for HTF trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 35:  # Need at least 35 bars for EMA34
        return np.zeros(n)
    
    # Calculate 12h EMA34 for HTF trend filter
    close_12h = df_12h['close'].values
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Calculate volume MA20 on 4h for confirmation
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for Donchian(20) and volume MA20
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_12h_aligned[i]) or np.isnan(volume_ma20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Calculate Donchian channels for current bar (using lookback of 20 periods)
        lookback_start = max(0, i - 19)
        highest_high = np.max(high[lookback_start:i+1])
        lowest_low = np.min(low[lookback_start:i+1])
        
        # Determine 12h HTF trend (bullish = price above EMA34)
        htf_12h_bullish = close[i] > ema_34_12h_aligned[i]
        htf_12h_bearish = close[i] < ema_34_12h_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 20-period MA
        volume_confirmed = volume[i] > 1.5 * volume_ma20[i]
        
        if position == 0:
            # Long setup: price breaks above upper Donchian + 12h uptrend + volume confirmation
            long_setup = (close[i] > highest_high) and htf_12h_bullish and volume_confirmed
            
            # Short setup: price breaks below lower Donchian + 12h downtrend + volume confirmation
            short_setup = (close[i] < lowest_low) and htf_12h_bearish and volume_confirmed
            
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
            # Exit: price breaks below lower Donchian (contrarian signal) OR 12h trend turns bearish
            if (close[i] < lowest_low) or (not htf_12h_bullish):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price breaks above upper Donchian (contrarian signal) OR 12h trend turns bullish
            if (close[i] > highest_high) or (htf_12h_bullish):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Donchian20_Breakout_12hEMA34_Trend_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0