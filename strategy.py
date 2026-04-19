#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d volume confirmation + 1w trend filter
# - Long when price breaks above 20-period Donchian high + 1d volume > 2x average + 1w close > 1w EMA(50)
# - Short when price breaks below 20-period Donchian low + 1d volume > 2x average + 1w close < 1w EMA(50)
# - Exit when price crosses back through Donchian midpoint or trend reversal
# - Position size: 0.25 to manage drawdown
# - Designed to capture strong trending moves with volume confirmation
# - Target: 20-40 trades/year to avoid excessive fee drag

name = "4h_Donchian20_1dVolume_1wTrend_v1"
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
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # 1d volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # 1w EMA(50) for trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # 4h Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Pre-compute trend condition: 1w close > 1w EMA50 for uptrend, < for downtrend
    trend_up = close > ema_50_1w_aligned
    trend_down = close < ema_50_1w_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Ensure enough data for all indicators (max of 20, 50)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(vol_ma_1d_aligned[i]) or np.isnan(ema_50_1w_aligned[i]) or \
           np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(donchian_mid[i]):
            signals[i] = 0.0
            continue
            
        # Volume filter: current 4h volume > 2x 1d average volume (scaled to 4h)
        # Approximate: 1 day = 6 * 4h bars, so compare to 1/6 of daily average
        vol_threshold = vol_ma_1d_aligned[i] / 6.0
        volume_filter = vol_threshold > 0 and volume[i] > 2.0 * vol_threshold
        
        if position == 0:
            # Look for long entry: Donchian breakout + uptrend + volume
            if close[i] > donchian_high[i] and trend_up[i] and volume_filter:
                signals[i] = 0.25
                position = 1
            # Look for short entry: Donchian breakdown + downtrend + volume
            elif close[i] < donchian_low[i] and trend_down[i] and volume_filter:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit on Donchian breakdown or trend reversal
            if close[i] < donchian_mid[i] or not trend_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit on Donchian breakout or trend reversal
            if close[i] > donchian_mid[i] or not trend_down[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals