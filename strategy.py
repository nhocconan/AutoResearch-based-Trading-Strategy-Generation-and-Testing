#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d volume spike and 1w trend filter
# - Long: price breaks above Donchian(20) high AND 1d volume > 2.0x 20-period average AND 1w close > 1w EMA200
# - Short: price breaks below Donchian(20) low AND 1d volume > 2.0x 20-period average AND 1w close < 1w EMA200
# - Exit: opposite Donchian breakout or volume drops below 1.5x average
# - Uses discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 20-50 trades/year (80-200 total over 4 years) to stay within fee drag limits
# - Donchian channels provide clear breakout levels with built-in stoploss
# - Volume confirmation ensures breakouts have conviction
# - Weekly trend filter prevents trading against the major trend
# - Works in both bull (breakouts with volume) and bear (breakdowns with volume) markets

name = "4h_1d_1w_donchian_volume_trend_v1"
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
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1d data ONCE before loop for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return signals
    
    # Pre-compute 1d volume confirmation (20-period average)
    volume_1d = df_1d['volume'].values
    volume_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    # Load 1w data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return signals
    
    # Pre-compute 1w EMA200 for trend filter
    ema200_1w = pd.Series(df_1w['close'].values).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # Pre-compute Donchian channels on 4h timeframe
    # Upper band: 20-period high
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    # Lower band: 20-period low
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(volume_sma_20_aligned[i]) or np.isnan(ema200_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        volume_current = volume[i]
        close_current = close[i]
        
        # Donchian breakout conditions
        breakout_high = close_current > donchian_high[i-1]  # Break above previous period's high
        breakout_low = close_current < donchian_low[i-1]    # Break below previous period's low
        
        # Volume confirmation: current volume > 2.0x 20-period average
        vol_confirm_strong = volume_current > 2.0 * volume_sma_20_aligned[i]
        vol_confirm_weak = volume_current > 1.5 * volume_sma_20_aligned[i]
        
        # Weekly trend filter
        weekly_uptrend = close_current > ema200_1w_aligned[i]
        weekly_downtrend = close_current < ema200_1w_aligned[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: breakout above Donchian high + strong volume confirmation + weekly uptrend
        if breakout_high and vol_confirm_strong and weekly_uptrend:
            enter_long = True
        
        # Short: breakout below Donchian low + strong volume confirmation + weekly downtrend
        if breakout_low and vol_confirm_strong and weekly_downtrend:
            enter_short = True
        
        # Exit conditions: opposite breakout or volume drops below weak threshold
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if break below Donchian low OR volume drops below weak threshold
            exit_long = breakout_low or (not vol_confirm_weak)
        elif position == -1:
            # Exit short if break above Donchian high OR volume drops below weak threshold
            exit_short = breakout_high or (not vol_confirm_weak)
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals