#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA34 trend filter and volume confirmation
# Uses discrete sizing 0.25 to balance return and drawdown. Target: 30-100 total trades over 4 years (7-25/year).
# Long when price breaks above Donchian(20) high AND weekly EMA34 uptrend AND volume spike.
# Short when price breaks below Donchian(20) low AND weekly EMA34 downtrend AND volume spike.
# Works in bull via breakout longs, in bear via breakdown shorts with trend filter.

name = "1d_Donchian20_1wEMA34_VolumeSpike_v1"
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
    
    # Calculate Donchian(20) channels
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1w EMA(34) for trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    ema_34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 34, 20)  # warmup
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_donchian_high = donchian_high[i]
        curr_donchian_low = donchian_low[i]
        curr_ema_34_1w = ema_34_1w_aligned[i]
        curr_volume_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike
            if curr_volume_spike:
                # Bullish entry: price breaks above Donchian high AND above weekly EMA34
                if (curr_close > curr_donchian_high and 
                    curr_close > curr_ema_34_1w):
                    signals[i] = 0.25
                    position = 1
                # Bearish entry: price breaks below Donchian low AND below weekly EMA34
                elif (curr_close < curr_donchian_low and 
                      curr_close < curr_ema_34_1w):
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position
            # Exit: price breaks below Donchian low OR loses weekly uptrend
            if (curr_close < curr_donchian_low or 
                curr_close < curr_ema_34_1w):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian high OR loses weekly downtrend
            if (curr_close > curr_donchian_high or 
                curr_close > curr_ema_34_1w):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals