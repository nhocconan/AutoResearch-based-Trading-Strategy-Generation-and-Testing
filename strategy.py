#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1d volume confirmation and 1w trend filter
# Donchian breakouts capture sustained momentum moves; 1d volume > 1.5x 20-period EMA confirms participation
# 1w EMA50 trend filter ensures we only trade in the direction of the weekly trend, reducing whipsaws in chop
# Discrete position sizing (0.25) controls fee drag while allowing meaningful exposure
# Target: 50-150 total trades over 4 years (12-37/year) for optimal risk-adjusted returns
# Works in bull markets by catching breakouts with trend, works in bear by only taking trend-aligned breaks
# Focus on BTC/ETH as primary symbols

name = "6h_Donchian20_Breakout_1dVolume_1wEMA50_Trend"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1d volume confirmation: volume > 1.5 x 20-period EMA
    vol_1d = df_1d['volume'].values
    vol_ema_20_1d = pd.Series(vol_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_confirmation_1d = vol_1d > (1.5 * vol_ema_20_1d)
    volume_confirmation_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_confirmation_1d)
    
    # 1w data for trend filter (EMA50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # 6h Donchian channels (20-period)
    lookback = 20
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(lookback - 1, n):
        highest_high[i] = np.max(high[i - lookback + 1:i + 1])
        lowest_low[i] = np.min(low[i - lookback + 1:i + 1])
    
    # Align Donchian levels to current bar (no look-ahead)
    # Since we're using completed 6h bars for calculation, no additional alignment needed
    # The values at index i are based on bars [i-lookback+1:i] which are all closed at bar i
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = max(50, lookback - 1)
    
    for i in range(start_idx, n):
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(volume_confirmation_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend bias from 1w EMA50
        bullish_trend = close_1w[-1] > ema_50_1w[-1] if len(close_1w) > 0 else False  # Current weekly trend
        # For bar i, we need the weekly trend as of the close of the weekly bar that contains bar i
        # Since we aligned the EMA, we can use the aligned value
        # Actually, better: use the weekly EMA value aligned to bar i to determine trend
        # But EMA is trend-following, so we need to know if price is above/below EMA
        # We'll use the 1w close vs EMA relationship, aligned
        # Re-calculate: get 1w close aligned
        close_1w_aligned = align_htf_to_ltf(prices, df_1w, close_1w)
        bullish_trend_aligned = close_1w_aligned > ema_50_1w_aligned
        
        if position == 0:  # Flat - look for new entries
            # Long: Close breaks above Donchian upper band with volume confirmation and bullish weekly trend
            if close[i] > highest_high[i] and volume_confirmation_1d_aligned[i] and bullish_trend_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Close breaks below Donchian lower band with volume confirmation and bearish weekly trend
            elif close[i] < lowest_low[i] and volume_confirmation_1d_aligned[i] and not bullish_trend_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Close drops below Donchian lower band (reversal) OR weekly trend turns bearish
            if close[i] < lowest_low[i] or not bullish_trend_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Close rises above Donchian upper band (reversal) OR weekly trend turns bullish
            if close[i] > highest_high[i] or bullish_trend_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals