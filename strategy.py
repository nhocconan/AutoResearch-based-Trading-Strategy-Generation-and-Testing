#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band Squeeze Breakout with 1d Volume Spike and 1w EMA200 Trend Filter
# Long when price breaks above upper BB(20,2) AND BBWidth at 20-period low AND volume > 2.0x 20-period median AND 1w EMA200 uptrend
# Short when price breaks below lower BB(20,2) AND BBWidth at 20-period low AND volume > 2.0x 20-period median AND 1w EMA200 downtrend
# Bollinger Squeeze identifies low volatility periods primed for breakout; volume spike confirms conviction; 1w EMA200 filters for higher-timeframe alignment.
# Works in bull markets (buy breakouts in uptrend) and bear markets (sell breakdowns in downtrend).
# Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years) to minimize fee drag.

name = "6h_BollingerSqueeze_Breakout_1wEMA200_Volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1w EMA200 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    ema_200_1w = pd.Series(df_1w['close'].values).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Calculate Bollinger Bands (20,2)
    bb_period = 20
    bb_std = 2.0
    sma_20 = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    std_20 = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper_bb = sma_20 + (bb_std * std_20)
    lower_bb = sma_20 - (bb_std * std_20)
    
    # Calculate BBWidth for squeeze detection
    bb_width = (upper_bb - lower_bb) / sma_20
    bb_width_rolling_min = pd.Series(bb_width).rolling(window=bb_period, min_periods=bb_period).min().values
    squeeze_condition = bb_width <= bb_width_rolling_min
    
    # Calculate 20-period volume median for volume confirmation
    vol_median_20 = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # track entry price for stoploss
    
    # Start after warmup for Bollinger Bands and volume
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(sma_20[i]) or 
            np.isnan(std_20[i]) or 
            np.isnan(ema_200_1w_aligned[i]) or 
            np.isnan(vol_median_20[i]) or 
            np.isnan(bb_width[i]) or 
            np.isnan(bb_width_rolling_min[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Trend filter: 1w EMA200 direction
        uptrend = curr_close > ema_200_1w_aligned[i]
        downtrend = curr_close < ema_200_1w_aligned[i]
        
        # Volume confirmation: current volume > 2.0x 20-period volume median
        if vol_median_20[i] <= 0 or np.isnan(vol_median_20[i]):
            volume_confirm = False
        else:
            volume_confirm = curr_volume > (vol_median_20[i] * 2.0)
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above upper BB AND squeeze AND volume spike AND uptrend
            if curr_high > upper_bb[i] and squeeze_condition[i] and volume_confirm and uptrend:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: price breaks below lower BB AND squeeze AND volume spike AND downtrend
            elif curr_low < lower_bb[i] and squeeze_condition[i] and volume_confirm and downtrend:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price closes below middle BB OR trend turns down
            if curr_close < sma_20[i] or not uptrend:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price closes above middle BB OR trend turns up
            if curr_close > sma_20[i] or not downtrend:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
    
    return signals