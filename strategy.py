#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with volume spike and 1w EMA34 trend filter
# Uses Donchian channel structure (20-period high/low) on daily timeframe
# Volume spike (>2.0x 20-period average) confirms institutional participation
# 1w EMA34 trend filter ensures trades align with higher timeframe momentum
# Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe
# Works in bull/bear: volume confirms breakout validity, 1w EMA34 filters counter-trend noise

name = "1d_Donchian20_VolumeSpike_1wEMA34_Trend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channel from previous 20 days
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 21:  # Need 20 days for Donchian + 1 for previous
        return np.zeros(n)
    
    # Previous 20 days' high/low for Donchian calculation (shifted by 1 to avoid look-ahead)
    prev_high_20 = pd.Series(df_1d['high']).rolling(window=20, min_periods=20).max().shift(1).values
    prev_low_20 = pd.Series(df_1d['low']).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Align daily levels to 1d timeframe (wait for daily bar to close)
    upper_20_aligned = align_htf_to_ltf(prices, df_1d, prev_high_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_1d, prev_low_20)
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    # 1w EMA34 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 35:  # Need 34 periods for EMA + 1 for previous
        return np.zeros(n)
    ema_34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(20, 34)  # warmup for volume MA and 1w EMA
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if np.isnan(upper_20_aligned[i]) or np.isnan(lower_20_aligned[i]) or np.isnan(vol_ma_20[i]) or np.isnan(ema_34_aligned[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_upper = upper_20_aligned[i]
        curr_lower = lower_20_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        curr_ema_34 = ema_34_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Only trade with volume confirmation and trend filter
            if curr_volume_confirm:
                # Bullish entry: price breaks above upper Donchian with volume and above 1w EMA34
                if curr_high > curr_upper and curr_close > curr_ema_34:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish entry: price breaks below lower Donchian with volume and below 1w EMA34
                elif curr_low < curr_lower and curr_close < curr_ema_34:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Exit when price breaks below lower Donchian (reversal signal)
            if curr_low < curr_lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when price breaks above upper Donchian (reversal signal)
            if curr_high > curr_upper:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals