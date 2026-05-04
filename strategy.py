#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Bollinger Band Squeeze Breakout + 1d Trend Filter + Volume Spike
# Bollinger Band squeeze (low volatility) precedes explosive moves. Breakout from squeeze
# with volume confirmation and 1d trend alignment captures strong moves in both bull and bear markets.
# Uses discrete position sizing (0.25) to minimize fee drag. Target: 20-50 trades/year on 4h.

name = "4h_BollingerSqueeze_1dTrend_VolumeBreakout"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for HTF trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Bollinger Bands (20, 2.0) on 4h
    bb_period = 20
    bb_std = 2.0
    ma = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    bb_std_dev = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper_bb = ma + (bb_std * bb_std_dev)
    lower_bb = ma - (bb_std * bb_std_dev)
    bb_width = (upper_bb - lower_bb) / ma  # Normalized bandwidth
    
    # Bollinger Band Squeeze: bandwidth below 20-period average bandwidth
    bb_width_ma = pd.Series(bb_width).rolling(window=20, min_periods=20).mean().values
    squeeze_condition = bb_width < bb_width_ma
    
    # Volume spike: volume > 1.5x 20-period volume EMA
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ema_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_aligned[i]) or np.isnan(ma[i]) or 
            np.isnan(upper_bb[i]) or np.isnan(lower_bb[i]) or 
            np.isnan(squeeze_condition[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: squeeze breakout above upper BB + 1d uptrend + volume spike
            if (close[i] > upper_bb[i] and 
                squeeze_condition[i-1] and  # Was squeezed before breakout
                close[i] > ema_50_aligned[i] and  # 1d uptrend
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: squeeze breakout below lower BB + 1d downtrend + volume spike
            elif (close[i] < lower_bb[i] and 
                  squeeze_condition[i-1] and  # Was squeezed before breakout
                  close[i] < ema_50_aligned[i] and  # 1d downtrend
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below middle BB OR 1d trend turns down
            if (close[i] < ma[i] or 
                close[i] < ema_50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above middle BB OR 1d trend turns up
            if (close[i] > ma[i] or 
                close[i] > ema_50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals