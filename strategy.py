#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Bollinger Band squeeze breakout + 1d EMA(34) trend filter + volume spike
# Long when price breaks above upper BB during low volatility (squeeze) + price > 1d EMA(34) + volume > 1.5x average
# Short when price breaks below lower BB during low volatility + price < 1d EMA(34) + volume > 1.5x average
# Exit when price returns to middle BB (mean reversion) or volatility expands (BB width > 20-period average)
# Designed for low trade frequency (20-40/year) to minimize fee drag. Works in both bull (breakouts) and bear (mean reversion in squeeze).

name = "4h_BBand_Squeeze_Breakout_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA(34) trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate EMA(34) on 1d for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA to 4h timeframe (wait for completed 1d bar)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Bollinger Bands (20, 2) on 4h
    bb_period = 20
    bb_std = 2.0
    sma = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    std = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper_band = sma + (bb_std * std)
    lower_band = sma - (bb_std * std)
    bb_width = upper_band - lower_band
    
    # Bollinger Band squeeze: width < 20-period average width (low volatility)
    bb_width_ma = pd.Series(bb_width).rolling(window=20, min_periods=20).mean().shift(1).values
    bb_squeeze = bb_width < bb_width_ma
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for all calculations)
    start_idx = 40  # max(20 for BB + 20 for BB width MA + 20 for vol MA +1 for shift)
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(sma[i]) or np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or 
            np.isnan(bb_squeeze[i]) or np.isnan(volume_spike[i]) or np.isnan(ema_34_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above upper BB during squeeze + price > 1d EMA(34) + volume spike
            long_condition = (close[i] > upper_band[i]) and bb_squeeze[i] and (close[i] > ema_34_1d_aligned[i]) and volume_spike[i]
            
            # Short entry: price breaks below lower BB during squeeze + price < 1d EMA(34) + volume spike
            short_condition = (close[i] < lower_band[i]) and bb_squeeze[i] and (close[i] < ema_34_1d_aligned[i]) and volume_spike[i]
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price returns to middle BB (mean reversion) OR volatility expands (BB width > 20-period average)
            if close[i] <= sma[i] or bb_width[i] > bb_width_ma[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price returns to middle BB (mean reversion) OR volatility expands (BB width > 20-period average)
            if close[i] >= sma[i] or bb_width[i] > bb_width_ma[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals