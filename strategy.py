#!/usr/bin/env python3
"""
6h_Momentum_RSI20_Breakout_VolumeSpike_1dTrend
Hypothesis: RSI(20) momentum on 6h timeframe combined with volume spikes and daily EMA trend filter captures strong momentum moves.
Works in bull markets by catching breakouts and in bear markets by shorting breakdowns with trend alignment.
Target: 20-40 trades/year on 6h timeframe with low trade frequency to minimize fee drag.
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
    
    # Get daily data for EMA trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily EMA(34) for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate RSI(20) on 6h close prices
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/20, adjust=False, min_periods=20).mean()
    avg_loss = loss.ewm(alpha=1/20, adjust=False, min_periods=20).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Volume spike: 2x 20-period average on 6h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 50  # Need enough data for RSI calculation
    
    for i in range(start_idx, n):
        if (np.isnan(rsi_values[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        rsi_val = rsi_values[i]
        ema_trend = ema_34_1d_aligned[i]
        
        if position == 0:
            # Long: RSI > 60 (momentum) with volume spike and price above daily EMA (uptrend)
            if rsi_val > 60 and volume_spike[i] and price > ema_trend:
                signals[i] = 0.25
                position = 1
            # Short: RSI < 40 (momentum down) with volume spike and price below daily EMA (downtrend)
            elif rsi_val < 40 and volume_spike[i] and price < ema_trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long position
            signals[i] = 0.25
            # Exit: RSI returns to neutral zone (40-60) or price breaks below daily EMA
            if rsi_val < 50 or price < ema_trend:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short position
            signals[i] = -0.25
            # Exit: RSI returns to neutral zone (40-60) or price breaks above daily EMA
            if rsi_val > 50 or price > ema_trend:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Momentum_RSI20_Breakout_VolumeSpike_1dTrend"
timeframe = "6h"
leverage = 1.0