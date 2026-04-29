#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d EMA200 trend filter and volume confirmation
# Elder Ray measures bull/bear power relative to EMA13: Bull Power = High - EMA13, Bear Power = Low - EMA13
# 1d EMA200 provides strong trend filter for regime alignment (bull/bear market adaptation)
# Volume > 2.0x average confirms institutional participation in breakouts
# Discrete position sizing (0.25) with exits on Elder Ray power reversal
# Designed for ~15-30 trades/year to minimize fee drag while capturing strong trending moves
# Works in bull/bear via trend filter: long only when 1d EMA200 up, short only when down

name = "6h_ElderRay_1dEMA200_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for EMA200 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d EMA200 for trend filter
    close_1d = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Calculate 6h EMA13 for Elder Ray ( Bull Power = High - EMA13, Bear Power = Low - EMA13 )
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Calculate 20-period average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 13, 200)  # Volume, EMA13, and 1d EMA200 warmup
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Skip if any required data is NaN
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema_200_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_bull_power = bull_power[i]
        curr_bear_power = bear_power[i]
        curr_ema200_1d = ema_200_1d_aligned[i]
        curr_vol_ma = vol_ma_20[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: Bull Power turns negative (bearish pressure)
            if curr_bull_power <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Bear Power turns positive (bullish pressure)
            if curr_bear_power >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Volume confirmation: current volume > 2.0x 20-period average
            vol_confirmed = curr_volume > 2.0 * curr_vol_ma
            
            # Long when Bull Power > 0, 1d EMA200 uptrend, volume confirmed
            if curr_bull_power > 0 and curr_close > curr_ema200_1d and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short when Bear Power < 0, 1d EMA200 downtrend, volume confirmed
            elif curr_bear_power < 0 and curr_close < curr_ema200_1d and vol_confirmed:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals