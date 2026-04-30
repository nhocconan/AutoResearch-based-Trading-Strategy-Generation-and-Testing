#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA34 trend filter and volume confirmation
# Uses Donchian channel from previous 20 days for structure
# Only trade breakouts above upper band or below lower band in direction of 1w EMA34 trend
# Volume spike (2.0x 20-day average) confirms institutional participation
# Works in bull markets via buying breakouts in uptrends and bear markets via selling breakdowns in downtrends
# Discrete sizing 0.25 minimizes fee churn. Target: 30-100 total trades over 4 years (7-25/year).

name = "1d_Donchian20_Breakout_1wEMA34_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop (MTF Rule #1)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 1w EMA34
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # warmup for Donchian and volume MA
    
    for i in range(start_idx, n):
        # Need prior 20 periods for Donchian calculation
        if i < 20:
            signals[i] = 0.0
            continue
            
        # Prior 20 periods' high/low for Donchian channel (excluding current bar)
        period_high = np.max(high[i-20:i])
        period_low = np.min(low[i-20:i])
        
        # Volume confirmation: volume > 2.0x 20-period average
        vol_ma_20 = np.mean(volume[max(0, i-20):i])
        volume_spike = volume[i] > (2.0 * vol_ma_20)
        
        curr_close = close[i]
        curr_ema_34_1w = ema_34_1w_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike
            if volume_spike:
                # Bullish entry: price breaks above upper Donchian band AND above 1w EMA34 (uptrend)
                if curr_close > period_high and curr_close > curr_ema_34_1w:
                    signals[i] = 0.25
                    position = 1
                # Bearish entry: price breaks below lower Donchian band AND below 1w EMA34 (downtrend)
                elif curr_close < period_low and curr_close < curr_ema_34_1w:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position
            # Exit when price falls below lower Donchian band or below 1w EMA34
            if curr_close < period_low or curr_close < curr_ema_34_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when price rises above upper Donchian band or above 1w EMA34
            if curr_close > period_high or curr_close > curr_ema_34_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals