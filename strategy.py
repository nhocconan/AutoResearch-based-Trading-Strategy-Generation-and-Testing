#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation
# Donchian channels provide robust price structure; breakouts indicate strong momentum
# In bull markets: buy when price breaks above upper channel with volume spike + price above 1d EMA34
# In bear markets: sell when price breaks below lower channel with volume spike + price below 1d EMA34
# Works in both regimes by using price channels as structure and volume for confirmation
# Target: 12-37 trades/year (50-150 total over 4 years) to minimize fee drag

name = "12h_Donchian20_1dEMA34_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d EMA(34) for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Donchian channel parameters
    donchian_period = 20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(donchian_period, n):  # Start from donchian_period to have valid indicators
        # Skip if any value is NaN
        if np.isnan(ema_34_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate Donchian channel using only historical data up to i-1
        lookback_start = max(0, i - donchian_period)
        lookback_end = i  # exclude current bar
        
        if lookback_end - lookback_start < donchian_period:
            # Not enough data for full lookback, use available
            period_high = np.max(high[lookback_start:lookback_end]) if lookback_end > lookback_start else high[i]
            period_low = np.min(low[lookback_start:lookback_end]) if lookback_end > lookback_start else low[i]
        else:
            period_high = np.max(high[lookback_start:lookback_end])
            period_low = np.min(low[lookback_start:lookback_end])
        
        # Volume confirmation: current volume > 2.0 x 20-period EMA of volume
        vol_lookback_start = max(0, i - 20)
        vol_lookback_end = i
        if vol_lookback_end - vol_lookback_start >= 20:
            vol_ema_20 = np.mean(volume[vol_lookback_start:vol_lookback_end])
        else:
            vol_ema_20 = np.mean(volume[:i]) if i > 0 else volume[i]
        
        volume_spike = volume[i] > (2.0 * vol_ema_20)
        
        # Donchian breakout signals with 1d trend filter
        # Long: price breaks above upper Donchian + volume spike + price above 1d EMA34
        # Short: price breaks below lower Donchian + volume spike + price below 1d EMA34
        if position == 0:
            if (close[i] > period_high and volume_spike and 
                close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            elif (close[i] < period_low and volume_spike and 
                  close[i] < ema_34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below lower Donchian (reversal) OR price below 1d EMA34
            if close[i] < period_low or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above upper Donchian (reversal) OR price above 1d EMA34
            if close[i] > period_high or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals