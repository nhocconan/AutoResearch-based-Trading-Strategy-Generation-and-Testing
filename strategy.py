#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h volume confirmation and 12h EMA50 trend filter
# Donchian breakouts capture sustained momentum moves; 12h volume > 1.3x 20-period EMA confirms institutional participation
# 12h EMA50 trend filter ensures we only trade in the direction of the intermediate trend, reducing whipsaws
# Discrete position sizing (0.25) controls fee drag while allowing meaningful exposure
# Target: 75-200 total trades over 4 years (19-50/year) for optimal risk-adjusted returns
# Works in bull markets by catching breakouts with trend, works in bear by only taking trend-aligned breaks
# Focus on BTC/ETH as primary symbols (SOL may benefit but not required)

name = "4h_Donchian20_Breakout_12hVolume_12hEMA50_Trend"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h data for volume confirmation and trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # 12h volume confirmation: volume > 1.3 x 20-period EMA
    vol_12h = df_12h['volume'].values
    vol_ema_20_12h = pd.Series(vol_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_confirmation_12h = vol_12h > (1.3 * vol_ema_20_12h)
    volume_confirmation_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_confirmation_12h)
    
    # 12h EMA50 trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # 4h Donchian channels (20-period)
    lookback = 20
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(lookback - 1, n):
        highest_high[i] = np.max(high[i - lookback + 1:i + 1])
        lowest_low[i] = np.min(low[i - lookback + 1:i + 1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = max(50, lookback - 1)
    
    for i in range(start_idx, n):
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(volume_confirmation_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend bias from 12h EMA50 (price vs EMA)
        bullish_trend_aligned = close_12h[-1] > ema_50_12h[-1] if len(close_12h) > 0 else False  # Current 12h trend
        # For bar i, we need the 12h trend as of the close of the 12h bar that contains bar i
        # Since we aligned the EMA, we can use the aligned values
        # Re-calculate: get 12h close aligned
        close_12h_aligned = align_htf_to_ltf(prices, df_12h, close_12h)
        bullish_trend_aligned = close_12h_aligned > ema_50_12h_aligned
        
        if position == 0:  # Flat - look for new entries
            # Long: Close breaks above Donchian upper band with volume confirmation and bullish 12h trend
            if close[i] > highest_high[i] and volume_confirmation_12h_aligned[i] and bullish_trend_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Close breaks below Donchian lower band with volume confirmation and bearish 12h trend
            elif close[i] < lowest_low[i] and volume_confirmation_12h_aligned[i] and not bullish_trend_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Close drops below Donchian lower band (reversal) OR 12h trend turns bearish
            if close[i] < lowest_low[i] or not bullish_trend_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Close rises above Donchian upper band (reversal) OR 12h trend turns bullish
            if close[i] > highest_high[i] or bullish_trend_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals