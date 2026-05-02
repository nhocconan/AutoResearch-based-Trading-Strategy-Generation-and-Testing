#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Bollinger Band squeeze breakout with 1d EMA50 trend filter and volume confirmation
# Bollinger Band width < 20th percentile indicates low volatility squeeze.
# Breakout above upper BB (long) or below lower BB (short) with 1d EMA50 trend alignment and volume spike
# captures explosive moves after consolidation. Works in both bull and bear markets by trading
# breakouts in the direction of the higher timeframe trend.
# Target: 20-50 trades/year (80-200 total over 4 years) on 4h timeframe.

name = "4h_BB_Squeeze_Breakout_1dEMA50_Trend_Volume_v1"
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
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Calculate EMA(50) on 1d for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Bollinger Bands (20, 2) on 4h
    bb_period = 20
    bb_std = 2
    sma_bb = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    bb_std_dev = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper_bb = sma_bb + (bb_std_dev * bb_std)
    lower_bb = sma_bb - (bb_std_dev * bb_std)
    bb_width = (upper_bb - lower_bb) / sma_bb  # Normalized width
    
    # Bollinger Band width percentile (20-period lookback) for squeeze detection
    bb_width_percentile = pd.Series(bb_width).rolling(window=50, min_periods=20).rank(pct=True).values
    squeeze_condition = bb_width_percentile < 0.20  # Below 20th percentile = squeeze
    
    # Volume confirmation (1.8x 20-period average) on 4h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for BB calculations)
    start_idx = 70  # max(50 for EMA, 50 for BB width percentile, 20 for volume +1 for shift)
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(squeeze_condition[i]) or 
            np.isnan(volume_spike[i]) or np.isnan(upper_bb[i]) or np.isnan(lower_bb[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: BB squeeze + price breaks above upper BB + 1d uptrend + volume spike
            if squeeze_condition[i] and close[i] > upper_bb[i] and close[i] > ema_50_1d_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short entry: BB squeeze + price breaks below lower BB + 1d downtrend + volume spike
            elif squeeze_condition[i] and close[i] < lower_bb[i] and close[i] < ema_50_1d_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price returns to middle BB (mean reversion) or trend reversal
            if close[i] < sma_bb[i] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price returns to middle BB (mean reversion) or trend reversal
            if close[i] > sma_bb[i] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals