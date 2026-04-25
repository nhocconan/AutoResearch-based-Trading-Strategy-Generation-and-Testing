#!/usr/bin/env python3
"""
1d Williams %R with 1w EMA50 Trend Filter and Volume Spike Confirmation
Hypothesis: Williams %R identifies overbought/oversold conditions. In trending markets (price above/below 1w EMA50),
we take mean-reversion entries at extremes. Volume spike (>2.0x 20-bar vol MA) confirms momentum.
Designed to work in both bull (buy dips in uptrend) and bear (sell rallies in downtrend) markets by adapting to regime.
Target: 15-25 trades/year to avoid fee drag while capturing strong reversals.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA50 trend filter (call ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 51:  # Need 50 for EMA + 1 for shift
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = pd.Series(df_1w['close'])
    ema_50_1w = close_1w.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Williams %R on 1d: (Highest High - Close) / (Highest High - Lowest Low) * -100
    # We'll use 14-period lookback
    lookback = 14
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    for i in range(lookback-1, n):
        highest_high[i] = np.max(high[i-lookback+1:i+1])
        lowest_low[i] = np.min(low[i-lookback+1:i+1])
    
    williams_r = np.full(n, np.nan)
    for i in range(lookback-1, n):
        if highest_high[i] != lowest_low[i]:
            williams_r[i] = ((highest_high[i] - close[i]) / (highest_high[i] - lowest_low[i])) * -100
        else:
            williams_r[i] = -50  # neutral when range is zero
    
    # Calculate 20-period volume MA for volume spike confirmation (1d)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Williams %R, EMA50, and volume MA
    start_idx = max(lookback, 51, 20)  # lookback for W%R, 51 for EMA50, 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(williams_r[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_50_val = ema_50_1w_aligned[i]
        wr = williams_r[i]
        vol_ma = vol_ma_20[i]
        
        # Trend filter: price above/below 1w EMA50
        price_above_ema = curr_close > ema_50_val
        price_below_ema = curr_close < ema_50_val
        
        # Volume confirmation: current volume > 2.0 * 20-period average
        volume_confirm = curr_volume > 2.0 * vol_ma
        
        # Williams %R conditions: oversold (< -80) or overbought (> -20)
        oversold = wr < -80
        overbought = wr > -20
        
        if position == 0:
            if price_above_ema:
                # Uptrend: look for oversold conditions to go long
                long_signal = oversold and volume_confirm
                if long_signal:
                    signals[i] = 0.25
                    position = 1
            elif price_below_ema:
                # Downtrend: look for overbought conditions to go short
                short_signal = overbought and volume_confirm
                if short_signal:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: Williams %R returns to neutral (> -50) or volume dries up
            if wr > -50 or not volume_confirm:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R returns to neutral (< -50) or volume dries up
            if wr < -50 or not volume_confirm:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Williams_%R_1wEMA50_Trend_VolumeSpike"
timeframe = "1d"
leverage = 1.0