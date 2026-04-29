#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and volume spike confirmation
# Long when price breaks above Donchian upper channel AND price > 12h EMA50 with volume > 2x 20-period average
# Short when price breaks below Donchian lower channel AND price < 12h EMA50 with volume spike
# Uses 12h EMA for trend direction (more responsive than daily) to catch trends earlier while avoiding chop
# Volume confirmation ensures breakouts have institutional participation
# Discrete position sizing (0.25) to minimize fee churn
# Target: 25-35 trades/year (100-140 total over 4 years) to stay within fee drag limits

name = "4h_Donchian20_12hEMA50_VolumeSpike_Trend_v3"
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
    ema_12h_50 = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h_50)
    
    # Donchian(20) channels on 4h
    donchian_window = 20
    upper_channel = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    lower_channel = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # warmup for EMA and Donchian
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if np.isnan(ema_12h_aligned[i]) or np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_upper = upper_channel[i]
        curr_lower = lower_channel[i]
        curr_ema = ema_12h_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Only trade with volume confirmation
            if curr_volume_confirm:
                # Bullish breakout: price breaks above upper Donchian channel AND above 12h EMA50 (uptrend)
                if curr_close > curr_upper and curr_close > curr_ema:
                    signals[i] = 0.25
                    position = 1
                # Bearish breakout: price breaks below lower Donchian channel AND below 12h EMA50 (downtrend)
                elif curr_close < curr_lower and curr_close < curr_ema:
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