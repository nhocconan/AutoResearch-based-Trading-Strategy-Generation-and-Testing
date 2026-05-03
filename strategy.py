#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d trend filter (EMA34) and 1w volume confirmation.
# Long when price breaks above 20-period Donchian high AND 1d close > 1d EMA34 (uptrend) AND 1w volume > 1.5x 20-period volume MA.
# Short when price breaks below 20-period Donchian low AND 1d close < 1d EMA34 (downtrend) AND 1w volume > 1.5x 20-period volume MA.
# Uses ATR-based stoploss (signal→0 when price moves against position by 2.5*ATR).
# Designed for 4h timeframe to achieve 75-200 total trades over 4 years (19-50/year) with tight entry conditions.
# Donchian channels provide objective breakout levels, 1d EMA34 filters for trend alignment, 1w volume confirms institutional participation.
# Works in both bull and bear markets by only trading breakouts in the direction of the 1d trend when volume confirms.

name = "4h_Donchian20_1dEMA34_1wVolumeSpike_ATR"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend direction
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Get 1w data for volume confirmation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 1w volume 20-period MA for spike detection
    volume_ma_1w = pd.Series(df_1w['volume']).rolling(window=20, min_periods=20).mean().values
    volume_ma_1w_aligned = align_htf_to_ltf(prices, df_1w, volume_ma_1w)
    
    # Calculate 4h Donchian channels (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate ATR(14) for stoploss
    tr1 = pd.Series(high - low).values
    tr2 = pd.Series(np.abs(high - np.roll(close, 1))).values
    tr3 = pd.Series(np.abs(low - np.roll(close, 1))).values
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_ma_1w_aligned[i]) or 
            np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        
        # Donchian breakout conditions
        breakout_up = high_val > high_roll[i]  # Price breaks above 20-period high
        breakout_down = low_val < low_roll[i]  # Price breaks below 20-period low
        
        # 1d trend conditions
        trend_up = close_val > ema_34_1d_aligned[i]   # 1d uptrend
        trend_down = close_val < ema_34_1d_aligned[i]  # 1d downtrend
        
        # 1w volume spike condition (using current 1w volume vs 20-period MA)
        # For proxy, we use current 4h volume scaled to weekly: sum of last ~28 4h bars (7 days * 4 bars/day)
        if i >= 28:
            vol_sum_28 = np.sum(volume[i-27:i+1])  # Approximate 1w volume
            vol_ma_1w_val = volume_ma_1w_aligned[i]
            volume_spike_1w = vol_sum_28 > (vol_ma_1w_val * 1.5)  # 1w volume > 1.5x 20-period MA
        else:
            volume_spike_1w = False
        
        if position == 0:
            # Long: Donchian breakout up AND 1d uptrend AND volume spike
            if breakout_up and trend_up and volume_spike_1w:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            # Short: Donchian breakout down AND 1d downtrend AND volume spike
            elif breakout_down and trend_down and volume_spike_1w:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
        elif position == 1:
            # Long position management
            # Stoploss: price < entry_price - 2.5 * atr[i]
            if close_val < entry_price - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Exit: price retouches Donchian mid-channel OR trend changes
            elif close_val < (high_roll[i] + low_roll[i]) / 2.0 or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Stoploss: price > entry_price + 2.5 * atr[i]
            if close_val > entry_price + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Exit: price retouches Donchian mid-channel OR trend changes
            elif close_val > (high_roll[i] + low_roll[i]) / 2.0 or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals