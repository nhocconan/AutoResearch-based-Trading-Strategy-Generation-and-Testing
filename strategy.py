#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d trend filter (EMA34) and 1d volume spike confirmation.
# Long when price breaks above 12h Donchian high AND 1d close > 1d EMA34 (uptrend) AND 1d volume > 1.5x 20-period volume MA.
# Short when price breaks below 12h Donchian low AND 1d close < 1d EMA34 (downtrend) AND 1d volume > 1.5x 20-period volume MA.
# Uses ATR-based stoploss (signal→0 when price moves against position by 2.5x ATR).
# Designed for 12h timeframe to achieve 50-150 total trades over 4 years (12-37/year) with strict entry conditions.
# Donchian channels provide objective breakout levels, 1d EMA34 filters for trend alignment, 1d volume confirms institutional participation.
# Works in both bull and bear markets by only trading breakouts in the direction of the 1d trend when volume confirms.

name = "12h_Donchian20_1dEMA34_1dVolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend and volume filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend direction
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d volume 20-period MA for spike detection
    volume_ma_1d = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    volume_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_1d)
    
    # Calculate 12h Donchian channels (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate ATR(14) for stoploss
    tr1 = pd.Series(high - low).values
    tr2 = pd.Series(np.abs(high - np.roll(close, 1))).values
    tr3 = pd.Series(np.abs(low - np.roll(close, 1))).values
    tr2[0] = tr1[0]  # First bar has no previous close
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_ma_1d_aligned[i]) or 
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
        
        # Volume spike condition (1d volume > 1.5x 20-period MA)
        volume_spike = df_1d['volume'].values[-1] > (volume_ma_1d_aligned[i] * 1.5) if len(df_1d) > 0 else False
        
        if position == 0:
            # Long: Donchian breakout up AND 1d uptrend AND volume spike
            if breakout_up and trend_up and volume_spike:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            # Short: Donchian breakout down AND 1d downtrend AND volume spike
            elif breakout_down and trend_down and volume_spike:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
        elif position == 1:
            # Long position management
            # Stoploss: price drops 2.5*ATR below entry
            if close_val <= entry_price - 2.5 * atr[i]:
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
            # Stoploss: price rises 2.5*ATR above entry
            if close_val >= entry_price + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Exit: price retouches Donchian mid-channel OR trend changes
            elif close_val > (high_roll[i] + low_roll[i]) / 2.0 or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals