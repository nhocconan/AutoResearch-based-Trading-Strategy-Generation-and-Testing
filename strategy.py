#!/usr/bin/env python3
# Hypothesis: 6h Bollinger Band squeeze breakout with 1d EMA50 trend filter and volume confirmation (>2.0x 20-bar avg). 
# BB squeeze (BBW < 20th percentile) indicates low volatility primed for expansion. Breakout direction filtered by 1d EMA50 trend. 
# Volume surge confirms institutional participation. Designed for BTC/ETH robustness: squeeze captures pre-move compression in both bull/bear regimes, 
# EMA50 filter ensures alignment with higher-timeframe trend, volume avoids false breakouts. Targets 12-37 trades/year on 6h timeframe.

name = "6h_BBandSqueeze_Breakout_1dEMA50_VolumeConfirm_v1"
timeframe = "6h"
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
    
    # Calculate Bollinger Bands (20, 2.0) on 6h
    bb_period = 20
    bb_std = 2.0
    sma_bb = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    bb_std_dev = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper_bb = sma_bb + bb_std * bb_std_dev
    lower_bb = sma_bb - bb_std * bb_std_dev
    bb_width = (upper_bb - lower_bb) / sma_bb  # normalized width
    
    # Calculate BBW percentile rank (lookback 100 bars) for squeeze detection
    bbw_rank = pd.Series(bb_width).rolling(window=100, min_periods=50).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    bbw_rank = np.where(np.isnan(bbw_rank), 50, bbw_rank)  # default to median if insufficient data
    
    # Squeeze condition: BBW < 20th percentile
    is_squeeze = bbw_rank < 20
    
    # Breakout conditions
    breakout_up = close > upper_bb
    breakout_down = close < lower_bb
    
    # Calculate 1d EMA50 for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # start after lookback
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(avg_volume[i]) or 
            np.isnan(bbw_rank[i]) or 
            np.isnan(upper_bb[i]) or 
            np.isnan(lower_bb[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: BB squeeze breakout up, price > 1d EMA50, volume spike (>2.0x avg)
            if (is_squeeze[i-1] and breakout_up[i] and  # squeeze was active previous bar
                close[i] > ema_50_1d_aligned[i] and 
                volume[i] > 2.0 * avg_volume[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: BB squeeze breakout down, price < 1d EMA50, volume spike (>2.0x avg)
            elif (is_squeeze[i-1] and breakout_down[i] and  # squeeze was active previous bar
                  close[i] < ema_50_1d_aligned[i] and 
                  volume[i] > 2.0 * avg_volume[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price reaches opposite BB (mean reversion) OR squeeze re-activates (low vol continuation)
            if (close[i] < sma_bb[i] or  # price back below SMA (mean reversion)
                is_squeeze[i]):          # volatility dropped again (squeeze re-activated)
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price reaches opposite BB (mean reversion) OR squeeze re-activates
            if (close[i] > sma_bb[i] or  # price back above SMA (mean reversion)
                is_squeeze[i]):          # volatility dropped again (squeeze re-activated)
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals