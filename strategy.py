#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 12h trend filter and volume confirmation.
# Long when price breaks above 20-period Donchian high AND 12h close > 12h EMA34 (uptrend) AND 4h volume > 1.5x 20-period volume MA.
# Short when price breaks below 20-period Donchian low AND 12h close < 12h EMA34 (downtrend) AND 4h volume > 1.5x 20-period volume MA.
# Uses ATR-based stoploss and session filter (08-20 UTC). Position size fixed at 0.25.
# Designed for 4h timeframe to achieve 75-200 total trades over 4 years (19-50/year) with strict entry conditions.
# Donchian channels provide objective breakout levels, 12h EMA34 filters for trend alignment, 4h volume confirms participation.
# Works in both bull and bear markets by only trading breakouts in the direction of the 12h trend when volume confirms.

name = "4h_Donchian20_12hEMA34_4hVolumeSpike_Session"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    # Calculate 12h EMA34 for trend direction
    ema_34_12h = pd.Series(df_12h['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Get 4h data for volume confirmation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4h volume 20-period MA for spike detection
    volume_ma_4h = pd.Series(df_4h['volume']).rolling(window=20, min_periods=20).mean().values
    volume_ma_4h_aligned = align_htf_to_ltf(prices, df_4h, volume_ma_4h)
    
    # Calculate ATR for stoploss (14-period)
    tr1 = pd.Series(high - low).values
    tr2 = pd.Series(np.abs(high - np.roll(close, 1))).values
    tr3 = pd.Series(np.abs(low - np.roll(close, 1))).values
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 4h Donchian channels (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_34_12h_aligned[i]) or np.isnan(volume_ma_4h_aligned[i]) or 
            np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or np.isnan(atr[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        atr_val = atr[i]
        
        # Donchian breakout conditions
        breakout_up = high_val > high_roll[i]  # Price breaks above 20-period high
        breakout_down = low_val < low_roll[i]  # Price breaks below 20-period low
        
        # 12h trend conditions
        trend_up = close_val > ema_34_12h_aligned[i]   # 12h uptrend
        trend_down = close_val < ema_34_12h_aligned[i]  # 12h downtrend
        
        # Volume spike condition (4h)
        volume_spike = volume[i] > (volume_ma_4h_aligned[i] * 1.5)  # 4h volume spike
        
        if position == 0:
            # Long: Donchian breakout up AND 12h uptrend AND volume spike AND session
            if breakout_up and trend_up and volume_spike:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            # Short: Donchian breakout down AND 12h downtrend AND volume spike AND session
            elif breakout_down and trend_down and volume_spike:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
        elif position == 1:
            # Exit conditions: ATR stoploss, trend reversal, or Donchian mid-channel retest
            if close_val < entry_price - 2.0 * atr_val:  # ATR stoploss
                signals[i] = 0.0
                position = 0
            elif close_val < (high_roll[i] + low_roll[i]) / 2.0 or not trend_up:  # Mid-channel or trend change
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit conditions: ATR stoploss, trend reversal, or Donchian mid-channel retest
            if close_val > entry_price + 2.0 * atr_val:  # ATR stoploss
                signals[i] = 0.0
                position = 0
            elif close_val > (high_roll[i] + low_roll[i]) / 2.0 or not trend_down:  # Mid-channel or trend change
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals