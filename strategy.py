#!/usr/bin/env python3
name = "12h_1dTrend_Volume_Squeeze_Breakout"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop for trend, range, and squeeze
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Daily EMA(34) for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Daily Bollinger Bands for squeeze detection
    bb_length = 20
    bb_mult = 2.0
    sma_bb = pd.Series(df_1d['close']).rolling(window=bb_length, min_periods=bb_length).mean().values
    bb_std = pd.Series(df_1d['close']).rolling(window=bb_length, min_periods=bb_length).std().values
    upper_bb = sma_bb + bb_mult * bb_std
    lower_bb = sma_bb - bb_mult * bb_std
    bb_width = (upper_bb - lower_bb) / sma_bb
    
    # Bollinger Band squeeze: width below 20-day percentile
    bb_width_ma = pd.Series(bb_width).rolling(window=20, min_periods=20).mean().values
    squeeze_condition = bb_width < bb_width_ma * 0.8
    
    # Align daily indicators to 12h timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    squeeze_aligned = align_htf_to_ltf(prices, df_1d, squeeze_condition)
    
    # 12-period high/low for breakout levels (1.5 days of 12h data)
    lookback = 12
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume confirmation: volume above 4-period average (2 days of 12h data)
    vol_ma_4 = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(lookback, 4, 34)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(squeeze_aligned[i]) or 
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(vol_ma_4[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: breakout above recent high during low volatility squeeze with volume and uptrend
            vol_condition = volume[i] > vol_ma_4[i] * 2.0
            uptrend = ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]
            
            if close[i] > highest_high[i] and squeeze_aligned[i] and vol_condition and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: breakdown below recent low during low volatility squeeze with volume and downtrend
            elif close[i] < lowest_low[i] and squeeze_aligned[i] and vol_condition and not uptrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: break below recent low or volatility expansion
            if close[i] < lowest_low[i] or not squeeze_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: break above recent high or volatility expansion
            if close[i] > highest_high[i] or not squeeze_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 12h Bollinger Squeeze Breakout with daily trend and volume confirmation
# - Bollinger Band squeeze (low volatility) precedes explosive moves
# - Breakout above 12-period high during squeeze + volume + daily uptrend = long
# - Breakdown below 12-period low during squeeze + volume + daily downtrend = short
# - Volume confirmation (2x average) ensures institutional participation
# - Works in bull markets (buy breakouts in uptrend) and bear markets (sell breakdowns in downtrend)
# - Exit when price breaks opposite boundary or volatility expands (squeeze ends)
# - Position size 0.25 targets ~20-50 trades/year to avoid fee drag
# - Uses daily trend filter to avoid whipsaws and align with higher timeframe bias
# - Bollinger squeeze identifies low-risk, high-reward setups before major moves
# - Designed for 12h timeframe to balance signal quality and trade frequency
# - Aims for 50-150 total trades over 4 years (12-37/year) to stay within limits