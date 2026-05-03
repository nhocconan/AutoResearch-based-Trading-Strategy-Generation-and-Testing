#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band squeeze breakout with 1d EMA34 trend filter and volume confirmation
# Bollinger Band squeeze (low volatility) precedes explosive moves in both bull and bear markets
# Breakout direction filtered by 1d EMA34 to trade with higher timeframe trend
# Volume spike confirms institutional participation
# Target: 12-30 trades/year (50-120 total over 4 years) to minimize fee drag
# Works in ranging markets (squeeze detection) and captures breakouts in trending markets

name = "6h_BollingerSqueeze_1dEMA34_VolumeSpike"
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
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d EMA(34) for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Bollinger Bands on 6h data
    bb_period = 20
    bb_std = 2.0
    sma_bb = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    std_bb = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper_bb = sma_bb + (bb_std * std_bb)
    lower_bb = sma_bb - (bb_std * std_bb)
    bb_width = (upper_bb - lower_bb) / sma_bb  # Normalized bandwidth
    
    # Bollinger Band squeeze: low volatility condition
    # Squeeze when BB width is below 20-period EMA of BB width (low volatility regime)
    bb_width_series = pd.Series(bb_width)
    bb_width_ema = bb_width_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    squeeze_condition = bb_width < bb_width_ema  # True when in low volatility squeeze
    
    # Volume confirmation: 20-period EMA on 6h volume
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start from 50 to have valid indicators
        # Skip if any value is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(sma_bb[i]) or 
            np.isnan(std_bb[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 2.0 x 20-period EMA (tight to avoid overtrading)
        volume_spike = volume[i] > (2.0 * vol_ema_20[i])
        
        # Bollinger Band breakout signals with 1d trend filter and squeeze condition
        # Long: price breaks above upper BB + squeeze condition + price above 1d EMA34 + volume spike
        # Short: price breaks below lower BB + squeeze condition + price below 1d EMA34 + volume spike
        if position == 0:
            if (close[i] > upper_bb[i] and squeeze_condition[i] and 
                close[i] > ema_34_1d_aligned[i] and volume_spike):
                signals[i] = 0.25
                position = 1
            elif (close[i] < lower_bb[i] and squeeze_condition[i] and 
                  close[i] < ema_34_1d_aligned[i] and volume_spike):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below middle BB (SMA) OR squeeze breaks (volatility expansion)
            if close[i] < sma_bb[i] or not squeeze_condition[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above middle BB (SMA) OR squeeze breaks (volatility expansion)
            if close[i] > sma_bb[i] or not squeeze_condition[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals