#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Bands breakout with 1d trend filter and volume confirmation
# Long when price breaks above upper BB(20,2) AND 1d close > 1d EMA50 AND volume > 2x 20-bar avg
# Short when price breaks below lower BB(20,2) AND 1d close < 1d EMA50 AND volume > 2x 20-bar avg
# Exit when price returns to middle BB(20) or volume drops
# Uses Bollinger Bands as dynamic support/resistance that adapts to volatility
# Works in both bull and bear markets by requiring 1d trend alignment to avoid counter-trend whipsaw
# Volume confirmation ensures breakouts have conviction
# Target: 12-37 trades/year via strict entry conditions reducing false breakouts

name = "6h_BollingerBreakout_1dEMA50_Trend_VolumeSpike_v1"
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
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:  # Need sufficient data for EMA50
        return np.zeros(n)
    
    # Calculate EMA(50) on 1d close
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Prepend zeros for alignment (since we lost first 49 bars in EMA calculation)
    ema_50_1d = np.concatenate([np.full(49, np.nan), ema_50_1d])
    
    # Align 1d EMA50 to 6h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Bollinger Bands on 6h close (20,2)
    close_s = pd.Series(close)
    bb_middle = close_s.rolling(window=20, min_periods=20).mean().values
    bb_std = close_s.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    
    # Volume confirmation: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Need sufficient history for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(bb_middle[i]) or 
            np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        price = close[i]
        bb_m = bb_middle[i]
        bb_u = bb_upper[i]
        bb_l = bb_lower[i]
        ema_50 = ema_50_1d_aligned[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long when price breaks above upper BB AND 1d close > 1d EMA50 AND volume confirmation
            if price > bb_u and close[i-1] <= bb_u and price > ema_50 and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below lower BB AND 1d close < 1d EMA50 AND volume confirmation
            elif price < bb_l and close[i-1] >= bb_l and price < ema_50 and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit when price returns to middle BB or volume drops
            if price <= bb_m or not vol_conf:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit when price returns to middle BB or volume drops
            if price >= bb_m or not vol_conf:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals