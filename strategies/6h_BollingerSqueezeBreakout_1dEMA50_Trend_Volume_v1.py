#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band Squeeze Breakout with 1d EMA50 trend filter and volume confirmation
# Bollinger Band Squeeze (BB Width < 20th percentile) identifies low volatility periods
# Breakout from squeeze + volume spike > 1.8x + 1d EMA50 trend filter captures explosive moves
# Works in bull/bear: squeeze breakouts occur in all regimes, volume confirmation ensures legitimacy
# Discrete sizing (0.25) targets 50-150 total trades over 4 years to avoid fee drag

name = "6h_BollingerSqueezeBreakout_1dEMA50_Trend_Volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Bollinger Bands (20, 2)
    bb_period = 20
    bb_std = 2
    close_s = pd.Series(close)
    bb_middle = close_s.rolling(window=bb_period, min_periods=bb_period).mean().values
    bb_std_dev = close_s.rolling(window=bb_period, min_periods=bb_period).std().values
    bb_upper = bb_middle + (bb_std * bb_std_dev)
    bb_lower = bb_middle - (bb_std * bb_std_dev)
    bb_width = bb_upper - bb_lower
    
    # Calculate BB Width percentile (20-period lookback for squeeze detection)
    bb_width_s = pd.Series(bb_width)
    bb_width_percentile = bb_width_s.rolling(window=50, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else np.nan, raw=False
    ).values
    
    # Squeeze condition: BB Width < 20th percentile
    squeeze = bb_width_percentile < 0.20
    
    # Breakout condition: price breaks above upper OR below lower band
    breakout_up = close > bb_upper
    breakout_down = close < bb_lower
    
    # Volume confirmation: volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.8 * vol_ma_20)
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, bb_period, 20, 50)  # warmup
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(bb_width_percentile[i]) or 
            np.isnan(bb_upper[i]) or 
            np.isnan(bb_lower[i]) or
            np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_squeeze = squeeze[i]
        curr_breakout_up = breakout_up[i]
        curr_breakout_down = breakout_down[i]
        curr_volume_confirm = volume_confirm[i]
        curr_ema_50_1d = ema_50_1d_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Only trade after squeeze breakout with volume confirmation and trend filter
            if not curr_squeeze and curr_volume_confirm:  # squeeze just broke
                # Bullish breakout: price above upper band + above 1d EMA50
                if curr_breakout_up and curr_close > curr_ema_50_1d:
                    signals[i] = 0.25
                    position = 1
                # Bearish breakout: price below lower band + below 1d EMA50
                elif curr_breakout_down and curr_close < curr_ema_50_1d:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position
            # Exit: price closes below middle band or squeeze re-establishes
            if curr_close < bb_middle[i] or squeeze[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price closes above middle band or squeeze re-establishes
            if curr_close > bb_middle[i] or squeeze[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals