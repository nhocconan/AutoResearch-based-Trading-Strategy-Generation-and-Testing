#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume confirmation and 12h EMA50 trend filter
# Donchian(20) captures significant price channels - breakouts indicate strong momentum
# Volume confirmation (>1.5x average) ensures breakout legitimacy and reduces false signals
# 12h EMA50 provides intermediate-term trend filter to avoid counter-trend trades
# Works in bull/bear markets: breakouts occur in all regimes, volume confirms legitimacy, trend filter reduces whipsaws
# Target: 75-200 total trades over 4 years (19-50/year) to minimize fee drag while maintaining edge

name = "4h_Donchian20_Volume_12hEMA50_Trend_v1"
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
    
    # Calculate Donchian channels (20-period) from previous bar to avoid look-ahead
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_high_prev = np.roll(high_20, 1)
    donchian_low_prev = np.roll(low_20, 1)
    donchian_high_prev[0] = np.nan
    donchian_low_prev[0] = np.nan
    
    # Breakout conditions
    breakout_up = close > donchian_high_prev
    breakout_down = close < donchian_low_prev
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    # Calculate 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 20)  # warmup
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(donchian_high_prev[i]) or 
            np.isnan(donchian_low_prev[i]) or
            np.isnan(ema_50_12h_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_breakout_up = breakout_up[i]
        curr_breakout_down = breakout_down[i]
        curr_volume_confirm = volume_confirm[i]
        curr_ema_50_12h = ema_50_12h_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Only trade on breakout with volume confirmation and trend filter
            if curr_volume_confirm:
                # Bullish breakout: price above Donchian high + above 12h EMA50
                if curr_breakout_up and curr_close > curr_ema_50_12h:
                    signals[i] = 0.25
                    position = 1
                # Bearish breakout: price below Donchian low + below 12h EMA50
                elif curr_breakout_down and curr_close < curr_ema_50_12h:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position
            # Exit: price closes below Donchian low (reversal) or above Donchian high (take profit)
            if curr_close < donchian_low_prev[i] or curr_close > donchian_high_prev[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price closes above Donchian high (reversal) or below Donchian low (take profit)
            if curr_close > donchian_high_prev[i] or curr_close < donchian_low_prev[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals