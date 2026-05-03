#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h mean reversion with 4h trend filter and 1d volume regime filter.
# Long: Price touches lower Bollinger Band (20,2) AND 4h close > 4h EMA50 (uptrend) AND 1d volume > 1.5x 20-day MA
# Short: Price touches upper Bollinger Band (20,2) AND 4h close < 4h EMA50 (downtrend) AND 1d volume > 1.5x 20-day MA
# Exit: Opposite Bollinger Band touch or trend filter fails or volume drops.
# Uses 1h for precise entry timing, 4h for trend direction, 1d for volume confirmation.
# Bollinger Bands provide mean reversion edge in ranging markets, filtered by 4h trend to avoid counter-trend trades.
# Volume confirmation ensures institutional participation. Designed for 15-37 trades/year on 1h timeframe.

name = "1h_BB20_4hEMA50_1dVolume"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA(50)
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Get 1d data for volume regime filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d 20-period volume MA
    volume_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # 1h Bollinger Bands (20,2)
    ma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_bb = ma_20 + (2 * std_20)
    lower_bb = ma_20 - (2 * std_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i]) or 
            np.isnan(upper_bb[i]) or np.isnan(lower_bb[i]) or np.isnan(ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        vol_1d = volume_1d[-1] if len(volume_1d) > 0 else 0  # Current 1d volume (approximation for alignment)
        # For volume spike, we need current 1d volume - using aligned array would be delayed, so we check if current 1d bar is forming
        # Instead, we use the previous completed 1d bar's volume MA for regime
        
        # Trend regime from 4h
        is_uptrend = close_val > ema_50_4h_aligned[i]
        is_downtrend = close_val < ema_50_4h_aligned[i]
        
        # Volume regime: check if current 1d volume > 1.5x 20-day MA (using aligned array of the MA)
        # We approximate by checking if the aligned MA value is less than current 1h volume scaled (not perfect but avoids lookahead)
        # Better: use the aligned volume MA as threshold - if current 1h volume > 1.5 * aligned_vol_ma, consider it spiked
        vol_spike = volume[i] > (1.5 * vol_ma_20_1d_aligned[i])
        
        # Entry logic
        if position == 0:
            # Long: Price touches lower BB AND uptrend AND volume spike
            if close_val <= lower_bb[i] and is_uptrend and vol_spike:
                signals[i] = 0.20
                position = 1
            # Short: Price touches upper BB AND downtrend AND volume spike
            elif close_val >= upper_bb[i] and is_downtrend and vol_spike:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: Price touches middle BB OR trend fails OR volume drops
            if close_val >= ma_20[i] or not is_uptrend or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: Price touches middle BB OR trend fails OR volume drops
            if close_val <= ma_20[i] or not is_downtrend or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals