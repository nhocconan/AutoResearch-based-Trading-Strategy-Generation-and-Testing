#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout + 1w EMA34 trend filter + volume confirmation
# Long when: price breaks above Donchian(20) high AND price > 1w EMA34 AND volume > 2.0x avg
# Short when: price breaks below Donchian(20) low AND price < 1w EMA34 AND volume > 2.0x avg
# Uses discrete sizing (0.30) to balance capture and fee drag. Works in bull/bear via 1w trend filter.
# Timeframe: 1d (primary), HTF: 1w for EMA34 trend.

name = "1d_Donchian20_1wEMA34_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop for 1w EMA34 trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 1w EMA34
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate Donchian(20) channels
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # warmup for Donchian
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if np.isnan(ema_34_1w_aligned[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_donchian_high = donchian_high[i]
        curr_donchian_low = donchian_low[i]
        curr_ema_34_1w = ema_34_1w_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        # Handle position exits
        if position == 1:  # Long position
            # Exit when price breaks below Donchian low (reversal signal)
            if curr_low < curr_donchian_low:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
                
        elif position == -1:  # Short position
            # Exit when price breaks above Donchian high (reversal signal)
            if curr_high > curr_donchian_high:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
                
        else:  # Flat - look for new entries
            # Long entry: price breaks above Donchian high AND price > 1w EMA34 AND volume confirm
            if (curr_high > curr_donchian_high and
                curr_close > curr_ema_34_1w and
                curr_volume_confirm):
                signals[i] = 0.30
                position = 1
            # Short entry: price breaks below Donchian low AND price < 1w EMA34 AND volume confirm
            elif (curr_low < curr_donchian_low and
                  curr_close < curr_ema_34_1w and
                  curr_volume_confirm):
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
    
    return signals