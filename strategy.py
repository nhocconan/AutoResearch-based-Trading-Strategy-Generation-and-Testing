#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily Bollinger Band Breakout with Weekly Trend Filter and Volume Spike
# Uses Bollinger Bands (20, 2.0) on daily timeframe for mean-reversion entries
# Weekly EMA (50) provides trend filter to trade in direction of higher timeframe trend
# Volume confirmation (>2.0x average) ensures institutional participation
# Designed to work in both bull and bear markets by fading extremes in ranging markets
# and following trend in trending markets
# Target: 15-25 trades/year (60-100 total over 4 years) to minimize fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop for Bollinger Bands
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Bollinger Bands (20, 2.0) on daily close
    close_1d = df_1d['close'].values
    bb_length = 20
    bb_mult = 2.0
    
    # Calculate basis (SMA)
    basis = np.zeros_like(close_1d)
    for i in range(bb_length - 1, len(close_1d)):
        basis[i] = np.mean(close_1d[i - bb_length + 1:i + 1])
    
    # Calculate standard deviation
    dev = np.zeros_like(close_1d)
    for i in range(bb_length - 1, len(close_1d)):
        dev[i] = np.std(close_1d[i - bb_length + 1:i + 1])
    
    # Calculate upper and lower bands
    upper_bb = basis + bb_mult * dev
    lower_bb = basis - bb_mult * dev
    
    # Align Bollinger Bands to daily timeframe
    upper_bb_aligned = align_htf_to_ltf(prices, df_1d, upper_bb)
    lower_bb_aligned = align_htf_to_ltf(prices, df_1d, lower_bb)
    basis_aligned = align_htf_to_ltf(prices, df_1d, basis)
    
    # Load weekly data ONCE before loop for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA (50) on close
    close_1w = df_1w['close'].values
    ema_length = 50
    ema_1w = np.zeros_like(close_1w)
    if len(close_1w) >= ema_length:
        ema_1w[ema_length - 1] = np.mean(close_1w[:ema_length])
        for i in range(ema_length, len(close_1w)):
            ema_1w[i] = (close_1w[i] * (2 / (ema_length + 1))) + (ema_1w[i - 1] * (1 - (2 / (ema_length + 1))))
    
    # Align weekly EMA to daily timeframe
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Volume confirmation: volume > 2.0x average volume (20-period)
    vol_series = pd.Series(volume)
    avg_vol = vol_series.rolling(window=20, min_periods=20).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(50, 50)  # for Bollinger Bands and volume calculations
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(upper_bb_aligned[i]) or np.isnan(lower_bb_aligned[i]) or 
            np.isnan(basis_aligned[i]) or np.isnan(ema_1w_aligned[i]) or np.isnan(avg_vol[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        # Trend filter: price above/below weekly EMA determines bias
        bullish_bias = price > ema_1w_aligned[i]
        bearish_bias = price < ema_1w_aligned[i]
        
        if position == 0:
            # Long: price touches lower Bollinger Band with volume filter and bullish bias
            if price <= lower_bb_aligned[i] and vol > 2.0 * avg_vol[i] and bullish_bias:
                position = 1
                signals[i] = position_size
            # Short: price touches upper Bollinger Band with volume filter and bearish bias
            elif price >= upper_bb_aligned[i] and vol > 2.0 * avg_vol[i] and bearish_bias:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to Bollinger basis (mean reversion target)
            if price >= basis_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to Bollinger basis (mean reversion target)
            if price <= basis_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_Bollinger_Band_MeanReversion_WeeklyEMA_Volume"
timeframe = "1d"
leverage = 1.0