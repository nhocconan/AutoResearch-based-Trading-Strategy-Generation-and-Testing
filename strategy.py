#!/usr/bin/env python3
"""
1h Bollinger Band Squeeze Breakout with 4h Trend Filter and Volume Spike
Hypothesis: Bollinger Band squeeze (low volatility) precedes breakouts. Combined with 4h EMA50 trend filter and volume spike (>2.0x 20-bar vol MA) to capture strong momentum moves. Works in bull markets via upside breakouts and in bear markets via downside breakdowns. Uses session filter (08-20 UTC) to reduce noise. Target: 15-35 trades/year to avoid fee drag.
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
    
    # Get 4h data for EMA50 trend filter (call ONCE before loop)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 51:  # Need 50 for EMA + 1 for shift
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    close_4h = pd.Series(df_4h['close'])
    ema_50_4h = close_4h.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate Bollinger Bands (20, 2.0) on 1h
    bb_period = 20
    bb_std = 2.0
    bb_ma = np.full(n, np.nan)
    bb_upper = np.full(n, np.nan)
    bb_lower = np.full(n, np.nan)
    bb_width = np.full(n, np.nan)
    
    for i in range(bb_period - 1, n):
        bb_ma[i] = np.mean(close[i-bb_period+1:i+1])
        bb_std_dev = np.std(close[i-bb_period+1:i+1])
        bb_upper[i] = bb_ma[i] + bb_std_dev * bb_std
        bb_lower[i] = bb_ma[i] - bb_std_dev * bb_std
        bb_width[i] = (bb_upper[i] - bb_lower[i]) / bb_ma[i] if bb_ma[i] != 0 else np.nan
    
    # Bollinger Band Squeeze: width < 20-period average width * 0.5
    bb_width_ma = np.full(n, np.nan)
    width_ma_period = 20
    for i in range(width_ma_period - 1, n):
        bb_width_ma[i] = np.mean(bb_width[i-width_ma_period+1:i+1])
    
    bb_squeeze = bb_width < (bb_width_ma * 0.5)
    
    # Calculate 20-period volume MA for volume spike confirmation
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    # Start index: need enough for EMA50, BB, and volume MA
    start_idx = max(50, bb_period, width_ma_period, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(bb_ma[i]) or 
            np.isnan(bb_width_ma[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_50_val = ema_50_4h_aligned[i]
        vol_ma = vol_ma_20[i]
        
        # Volume confirmation: current volume > 2.0 * 20-period average
        volume_confirm = curr_volume > 2.0 * vol_ma
        
        # Trend filter: price above/below 4h EMA50
        price_above_ema = curr_close > ema_50_val
        price_below_ema = curr_close < ema_50_val
        
        if position == 0:
            # Long: BB squeeze breakout above upper band + price above 4h EMA50 + volume confirmation
            long_signal = bb_squeeze[i] and (curr_high > bb_upper[i]) and price_above_ema and volume_confirm
            # Short: BB squeeze breakout below lower band + price below 4h EMA50 + volume confirmation
            short_signal = bb_squeeze[i] and (curr_low < bb_lower[i]) and price_below_ema and volume_confirm
            
            if long_signal:
                signals[i] = 0.20
                position = 1
            elif short_signal:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price crosses back below upper band OR price crosses below 4h EMA50
            if (curr_close < bb_upper[i]) or (curr_close < ema_50_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price crosses back above lower band OR price crosses above 4h EMA50
            if (curr_close > bb_lower[i]) or (curr_close > ema_50_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_BollingerSqueeze_Breakout_4hEMA50_Trend_VolumeSpike"
timeframe = "1h"
leverage = 1.0