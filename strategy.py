#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation.
# Long when price breaks above 4h Donchian upper band with 1d uptrend (price > 1d EMA50) and volume spike (>1.8x 20-bar avg).
# Short when price breaks below 4h Donchian lower band with 1d downtrend (price < 1d EMA50) and volume spike.
# Exit on opposite 4h Donchian band touch (mean reversion within the channel).
# Uses 4h for signal direction and structure, 1h only for entry timing precision.
# Session filter (08-20 UTC) reduces noise trades. Target: 15-37 trades/year per symbol.
# Position size: 0.20 (discrete level to minimize fee churn).

name = "1h_Donchian20_1dEMA50_Trend_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session filter (08-20 UTC) ONCE before loop
    if isinstance(prices.index, pd.DatetimeIndex):
        hours = prices.index.hour
    else:
        hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 4h data ONCE before loop for Donchian structure
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4h Donchian(20) from completed 4h bars
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    donchian_high_4h = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low_4h = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align 4h Donchian levels to 1h timeframe (completed 4h bar only)
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high_4h)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low_4h)
    
    # Load 1d data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: volume > 1.8x 20-period average (tight to avoid overtrading)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.8 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for EMA50 and Donchian
    
    for i in range(start_idx, n):
        # Skip if not in trading session or indicators not available
        if not in_session[i] or \
           (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_donch_high = donchian_high_aligned[i]
        curr_donch_low = donchian_low_aligned[i]
        curr_ema_50_1d = ema_50_1d_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above Donchian high, uptrend (price > 1d EMA50), volume spike
            if (curr_close > curr_donch_high and 
                curr_close > curr_ema_50_1d and 
                curr_volume_confirm):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below Donchian low, downtrend (price < 1d EMA50), volume spike
            elif (curr_close < curr_donch_low and 
                  curr_close < curr_ema_50_1d and 
                  curr_volume_confirm):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:  # Long position
            # Exit condition: price touches Donchian low (mean reversion)
            if curr_close <= curr_donch_low:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit condition: price touches Donchian high (mean reversion)
            if curr_close >= curr_donch_high:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals