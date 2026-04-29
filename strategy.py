#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation
# Long when price breaks above Donchian upper channel in bullish regime (price > 12h EMA50) with volume spike
# Short when price breaks below Donchian lower channel in bearish regime (price < 12h EMA50) with volume spike
# Uses 12h EMA50 to filter for trending markets, avoiding whipsaws in ranging conditions
# Volume confirmation ensures breakouts have institutional participation
# Target: 25-35 trades/year (100-140 total over 4 years) to minimize fee drag

name = "4h_Donchian20_12hEMA50_VolumeSpike_Trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop for 12h calculations
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA(50) for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 12h EMA50 to 4h timeframe (completed 12h bar only)
    ema_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Donchian(20) channels on 4h
    donchian_window = 20
    upper_channel = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    lower_channel = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(55, 20)  # warmup for EMA and Donchian
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if np.isnan(ema_aligned[i]) or np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_upper = upper_channel[i]
        curr_lower = lower_channel[i]
        curr_ema = ema_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        # Trend filter: only trade in direction of 12h EMA50
        is_bullish = curr_close > curr_ema
        is_bearish = curr_close < curr_ema
        
        if position == 0:  # Flat - look for new entries
            # Only trade with volume confirmation and in trending regime
            if curr_volume_confirm:
                # Bullish breakout: price breaks above upper Donchian channel in bullish regime
                if is_bullish and curr_close > curr_upper:
                    signals[i] = 0.25
                    position = 1
                # Bearish breakout: price breaks below lower Donchian channel in bearish regime
                elif is_bearish and curr_close < curr_lower:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position - exit conditions
            # Exit when: price returns to middle of channel OR breaks below lower channel with volume
            middle_channel = (curr_upper + curr_lower) / 2.0
            
            if curr_close <= middle_channel or (curr_close < curr_lower and curr_volume_confirm):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position - exit conditions
            # Exit when: price returns to middle of channel OR breaks above upper channel with volume
            middle_channel = (curr_upper + curr_lower) / 2.0
            
            if curr_close >= middle_channel or (curr_close > curr_upper and curr_volume_confirm):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals