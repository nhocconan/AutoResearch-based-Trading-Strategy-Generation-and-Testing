#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA34 trend filter and volume confirmation.
# Uses 1d timeframe to minimize fee drag, targeting 30-100 trades over 4 years.
# Long when price breaks above 20-day high with 1w EMA34 uptrend and volume spike.
# Short when price breaks below 20-day low with 1w EMA34 downtrend and volume spike.
# Designed to work in both bull and bear markets via 1w EMA34 trend filter.
# Volume confirmation uses 1.5x 20-day average to reduce false signals.

name = "1d_Donchian20_1wEMA34_VolumeConfirm_v1"
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
    
    # Load 1w data ONCE before loop for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate 1w EMA34 for trend filter
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 34  # warmup for EMA34
    
    for i in range(start_idx, n):
        # Skip if indicators not available
        if np.isnan(ema_34_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema_34_1w = ema_34_1w_aligned[i]
        
        # Calculate Donchian channels using previous 20 days (completed)
        if i >= 20:
            donchian_high = np.max(high[i-20:i])
            donchian_low = np.min(low[i-20:i])
        else:
            donchian_high = np.nan
            donchian_low = np.nan
        
        # Volume confirmation: volume > 1.5x 20-day average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-20:i])
            volume_confirm = volume[i] > (1.5 * vol_ma_20)
        else:
            volume_confirm = False
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above Donchian high, 1w EMA34 uptrend, volume confirmation
            if (not np.isnan(donchian_high) and 
                curr_close > donchian_high and 
                curr_close > curr_ema_34_1w and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: price breaks below Donchian low, 1w EMA34 downtrend, volume confirmation
            elif (not np.isnan(donchian_low) and 
                  curr_close < donchian_low and 
                  curr_close < curr_ema_34_1w and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        
        elif position == 1:  # Long position
            # Exit: price breaks below Donchian low or reverses below entry
            if (not np.isnan(donchian_low) and curr_close < donchian_low) or curr_close < entry_price:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian high or reverses above entry
            if (not np.isnan(donchian_high) and curr_close > donchian_high) or curr_close > entry_price:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals