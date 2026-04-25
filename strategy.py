#!/usr/bin/env python3
"""
1h Bollinger Band Squeeze Breakout with 4h EMA50 Trend Filter and Volume Spike
Hypothesis: Bollinger Band squeeze (low volatility) followed by breakout captures explosive moves.
Using 4h EMA50 for trend filter ensures alignment with higher timeframe momentum.
Volume spike confirms institutional participation. Designed for 1h timeframe targeting 15-37 trades/year.
Works in bull/bear regimes via trend filter - only takes breakouts in direction of 4h trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 4h data ONCE before loop for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # 4h EMA50 for trend filter
    ema_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Bollinger Bands (20, 2) on 1h
    bb_period = 20
    bb_std = 2
    sma = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    std = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper_band = sma + (bb_std * std)
    lower_band = sma - (bb_std * std)
    bb_width = (upper_band - lower_band) / sma  # normalized bandwidth
    
    # Bollinger Band Squeeze: bandwidth below 20-period mean bandwidth
    bb_width_ma = pd.Series(bb_width).rolling(window=20, min_periods=20).mean().values
    squeeze = bb_width < bb_width_ma
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for calculations
    start_idx = max(bb_period, 20, 50)  # BB, BB width MA, EMA50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(sma[i]) or np.isnan(std[i]) or 
            np.isnan(bb_width_ma[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        vol_spike = volume_spike[i]
        is_squeeze = squeeze[i]
        
        # Trend filter: price relative to 4h EMA50
        bullish_bias = curr_close > ema_4h_aligned[i]
        bearish_bias = curr_close < ema_4h_aligned[i]
        
        if position == 0:
            # Look for entry signals - require squeeze breakout + trend + volume
            # Long: price breaks above upper band AFTER squeeze, bullish bias, volume spike
            long_entry = is_squeeze and (curr_high > upper_band[i]) and bullish_bias and vol_spike
            # Short: price breaks below lower band AFTER squeeze, bearish bias, volume spike
            short_entry = is_squeeze and (curr_low < lower_band[i]) and bearish_bias and vol_spike
            
            if long_entry:
                signals[i] = 0.20
                position = 1
            elif short_entry:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: price falls below middle band (mean reversion) OR loss of bullish bias
            if (curr_close < sma[i]) or (curr_close < ema_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short position management
            # Exit: price rises above middle band (mean reversion) OR loss of bearish bias
            if (curr_close > sma[i]) or (curr_close > ema_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_BollingerSqueeze_Breakout_4hEMA50_Trend_VolumeSpike"
timeframe = "1h"
leverage = 1.0