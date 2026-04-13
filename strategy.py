#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Bollinger Band Squeeze with 1d Trend Filter and Volume Spike
# Bollinger Band squeeze (low volatility) precedes explosive moves.
# Combine with 1d trend direction (EMA50) to filter direction and volume spike for confirmation.
# Works in bull/bear markets: squeeze breakouts capture volatility expansion in any regime.
# Target: 20-40 trades per year (80-160 total over 4 years) for 4h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1-day data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # EMA(50) for 1d trend filter
    ema50_1d = np.zeros(len(close_1d))
    ema_multiplier = 2 / (50 + 1)
    ema50_1d[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        ema50_1d[i] = (close_1d[i] - ema50_1d[i-1]) * ema_multiplier + ema50_1d[i-1]
    
    # Align 1d EMA to 4h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Bollinger Bands on 4h (20, 2)
    bb_length = 20
    bb_mult = 2.0
    
    basis = np.full(n, np.nan)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    # Calculate SMA basis
    for i in range(bb_length - 1, n):
        basis[i] = np.mean(close[i - bb_length + 1:i + 1])
    
    # Calculate standard deviation
    dev = np.full(n, np.nan)
    for i in range(bb_length - 1, n):
        dev[i] = np.std(close[i - bb_length + 1:i + 1])
    
    # Upper and lower bands
    upper = basis + bb_mult * dev
    lower = basis - bb_mult * dev
    
    # Bollinger Band Width (normalized)
    bb_width = np.full(n, np.nan)
    for i in range(bb_length - 1, n):
        if basis[i] != 0:
            bb_width[i] = (upper[i] - lower[i]) / basis[i]
    
    # Average volume (20-period = 10 hours) for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(bb_length - 1, n):
        # Skip if any required data is not ready
        if (np.isnan(basis[i]) or np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(bb_width[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        bbw = bb_width[i]
        ema_trend = ema50_1d_aligned[i]
        
        # Bollinger Band Squeeze: BB width below 20-period average
        bbw_ma = np.full(n, np.nan)
        if i >= 20 + bb_length - 1:
            bbw_ma[i] = np.mean(bb_width[i-20:i])
        
        squeeze = bbw < bbw_ma[i] if not np.isnan(bbw_ma[i]) else False
        
        # Volume confirmation: current volume > 2.0x average volume
        volume_confirm = vol > 2.0 * avg_vol
        
        if position == 0:
            # Long: Squeeze breakout above upper band + above 1d EMA50 + volume
            if (price > upper[i] and 
                price > ema_trend and
                squeeze and
                volume_confirm):
                position = 1
                signals[i] = position_size
            # Short: Squeeze breakout below lower band + below 1d EMA50 + volume
            elif (price < lower[i] and 
                  price < ema_trend and
                  squeeze and
                  volume_confirm):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Price breaks below basis (middle band) or squeeze ends
            if (price < basis[i] or not squeeze):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Price breaks above basis (middle band) or squeeze ends
            if (price > basis[i] or not squeeze):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_BollingerSqueeze_Trend_Volume"
timeframe = "4h"
leverage = 1.0