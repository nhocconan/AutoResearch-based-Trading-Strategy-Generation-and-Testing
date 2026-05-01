#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h trend filter and volume confirmation.
# Uses 12h EMA50 as trend filter to avoid counter-trend trades.
# Long: Donchian breakout above upper channel + price > 12h EMA50 + volume spike.
# Short: Donchian breakdown below lower channel + price < 12h EMA50 + volume spike.
# Exits on opposite Donchian touch (mean reversion within channel).
# Discrete position sizing 0.25 balances return and drawdown.
# Target: 75-200 trades over 4 years (19-50/year) to minimize fee drag.

name = "4h_Donchian20_Breakout_12hEMA50_Trend_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # 4h Donchian(20) channels
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 2.0 * 20-period average volume on 4h
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = max(50, 20) + 1  # 51 (for EMA50 and Donchian20)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(donchian_high[i]) or
            np.isnan(donchian_low[i]) or
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            if position != 0:
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Trend filter: 12h EMA50 direction
        uptrend = curr_close > ema_50_12h_aligned[i]
        downtrend = curr_close < ema_50_12h_aligned[i]
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        # Donchian breakout conditions
        breakout_up = curr_high > donchian_high[i]  # Break above upper Donchian
        breakdown_down = curr_low < donchian_low[i]  # Break below lower Donchian
        
        # Donchian reversion conditions (exit signals)
        reversion_up = curr_low > donchian_low[i]   # Back above lower channel (exit short)
        reversion_down = curr_high < donchian_high[i]  # Below upper channel (exit long)
        
        if position == 0:  # Flat - look for new entries
            # Long: Donchian breakout up AND uptrend AND volume confirmation
            if breakout_up and uptrend and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Donchian breakdown down AND downtrend AND volume confirmation
            elif breakdown_down and downtrend and vol_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit when price re-enters the Donchian channel (mean reversion)
            if reversion_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when price re-enters the Donchian channel (mean reversion)
            if reversion_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals