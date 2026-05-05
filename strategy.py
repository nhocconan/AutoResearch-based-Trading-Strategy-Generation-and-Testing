#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d volume spike and ADX regime filter
# Long when: price breaks above 20-period Donchian high AND 1d volume > 1.5x 20-period average AND 1d ADX > 25
# Short when: price breaks below 20-period Donchian low AND 1d volume > 1.5x 20-period average AND 1d ADX > 25
# Exit when price retouches the Donchian midpoint (mean reversion within the channel)
# Uses 12h timeframe with 1d HTF for volume and ADX regime filters (target: 50-150 total over 4 years)
# Donchian channels provide clear trend structure and breakout signals
# Volume confirmation ensures breakouts have conviction
# 1d ADX > 25 filters for trending markets, avoiding whipsaws in ranges
# Discrete position sizing (0.25) minimizes fee churn while maintaining adequate exposure

name = "12h_Donchian20_1dVolumeSpike_ADX_Regime"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data ONCE before loop for volume and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough for ADX and volume MA calculation
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d ADX(14)
    if len(high_1d) >= 14:
        # True Range
        tr1 = np.abs(np.diff(high_1d))
        tr2 = np.abs(np.diff(low_1d))
        tr3 = np.abs(np.diff(close_1d))
        tr = np.maximum(np.maximum(tr1, tr2), tr3)
        tr = np.concatenate([[np.nan], tr])  # First value is NaN
        
        # Directional Movement
        up_move = np.diff(high_1d)
        down_move = -np.diff(low_1d)  # Negative because we want positive values when low decreases
        up_move = np.concatenate([[np.nan], up_move])
        down_move = np.concatenate([[np.nan], down_move])
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Smoothed values using Wilder's smoothing (equivalent to EMA with alpha=1/period)
        def wilders_smoothing(data, period):
            result = np.full_like(data, np.nan)
            if len(data) < period:
                return result
            # First value is simple average
            result[period-1] = np.nanmean(data[:period])
            # Subsequent values: smoothed = previous * (1 - 1/period) + current * (1/period)
            alpha = 1 / period
            for i in range(period, len(data)):
                if not np.isnan(data[i]):
                    result[i] = result[i-1] * (1 - alpha) + data[i] * alpha
                else:
                    result[i] = result[i-1]
            return result
        
        tr14 = wilders_smoothing(tr, 14)
        plus_dm14 = wilders_smoothing(plus_dm, 14)
        minus_dm14 = wilders_smoothing(minus_dm, 14)
        
        # Avoid division by zero
        plus_di14 = np.where(tr14 != 0, (plus_dm14 / tr14) * 100, 0)
        minus_di14 = np.where(tr14 != 0, (minus_dm14 / tr14) * 100, 0)
        
        dx = np.where((plus_di14 + minus_di14) != 0, 
                      np.abs(plus_di14 - minus_di14) / (plus_di14 + minus_di14) * 100, 0)
        adx = wilders_smoothing(dx, 14)
    else:
        adx = np.full(len(close_1d), np.nan)
    
    # Calculate 1d volume 20-period EMA for volume spike detection
    volume_ema_20_1d = pd.Series(volume_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align HTF indicators to 12h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    volume_ema_aligned = align_htf_to_ltf(prices, df_1d, volume_ema_20_1d)
    
    # Calculate 12h Donchian channels (20-period)
    def donchian_high(data, period):
        return pd.Series(data).rolling(window=period, min_periods=period).max().values
    
    def donchian_low(data, period):
        return pd.Series(data).rolling(window=period, min_periods=period).min().values
    
    def donchian_mid(data, period):
        high = donchian_high(data, period)
        low = donchian_low(data, period)
        return (high + low) / 2
    
    donchian_high_20 = donchian_high(high, 20)
    donchian_low_20 = donchian_low(low, 20)
    donchian_mid_20 = donchian_mid(close, 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(adx_aligned[i]) or np.isnan(volume_ema_aligned[i]) or 
            np.isnan(donchian_high_20[i]) or np.isnan(donchian_low_20[i]) or 
            np.isnan(donchian_mid_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian high AND volume spike AND ADX > 25
            if (close[i] > donchian_high_20[i] and 
                volume_1d[-1] > 1.5 * volume_ema_20_1d[-1] if len(volume_1d) > 0 else False and  # Use latest 1d volume
                adx_aligned[i] > 25):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low AND volume spike AND ADX > 25
            elif (close[i] < donchian_low_20[i] and 
                  volume_1d[-1] > 1.5 * volume_ema_20_1d[-1] if len(volume_1d) > 0 else False and  # Use latest 1d volume
                  adx_aligned[i] > 25):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price retouches Donchian midpoint
            if close[i] <= donchian_mid_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price retouches Donchian midpoint
            if close[i] >= donchian_mid_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals