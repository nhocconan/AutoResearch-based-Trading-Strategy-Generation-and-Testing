#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA34 trend filter and volume confirmation.
# Long when price breaks above upper Donchian channel AND price > 1w EMA34 AND volume > 2.0x 20-bar average.
# Short when price breaks below lower Donchian channel AND price < 1w EMA34 AND volume > 2.0x 20-bar average.
# Exit when price crosses the middle Donchian channel (20-bar midpoint).
# Uses discrete position sizing (0.25) to minimize fee drag and control drawdown.
# Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe.
# Works in bull/bear via 1w EMA34 trend filter and strict volume confirmation to avoid false breakouts.
# Primary timeframe: 1d, HTF: 1w for trend filter.

name = "1d_Donchian20_1wEMA34_Trend_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 1w EMA34 for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate Donchian channels from 20-period high/low
    high_ma_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_ma_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    upper_channel = high_ma_20
    lower_channel = low_ma_20
    middle_channel = (upper_channel + lower_channel) / 2.0
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # warmup for EMA and Donchian channels
    
    for i in range(start_idx, n):
        # Skip if indicators not available
        if (np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or np.isnan(middle_channel[i]) or 
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: break above upper channel, uptrend (price > 1w EMA34), volume confirmation
            if (curr_high > upper_channel[i] and 
                curr_close > ema_34_1w_aligned[i] and 
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: break below lower channel, downtrend (price < 1w EMA34), volume confirmation
            elif (curr_low < lower_channel[i] and 
                  curr_close < ema_34_1w_aligned[i] and 
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position
            # Exit condition: price crosses below middle channel
            if curr_close < middle_channel[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit condition: price crosses above middle channel
            if curr_close > middle_channel[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals