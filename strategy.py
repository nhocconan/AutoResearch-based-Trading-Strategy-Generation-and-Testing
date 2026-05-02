#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout + 1w HMA(21) trend + volume confirmation (2x avg volume)
# Uses 1d primary timeframe for Donchian breakout signals
# 1w HMA21 confirms longer-term trend direction (avoids counter-trend trades)
# Volume confirmation (2.0x 20-period average) ensures strong participation
# Discrete position sizing (0.25) balances profit potential with fee drag minimization
# Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe
# Donchian provides clear structure, 1w HMA adds robust trend filter, volume confirms conviction
# Works in both bull and bear markets by only trading in direction of 1w trend

name = "1d_Donchian20_1wHMA21_Trend_Volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for HMA21 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:
        return np.zeros(n)
    
    # Calculate 1w HMA(21)
    close_1w = pd.Series(df_1w['close'])
    half_len = int(21 / 2)
    sqrt_len = int(np.sqrt(21))
    
    # WMA function
    def wma(values, window):
        weights = np.arange(1, window + 1)
        return np.convolve(values, weights, mode='valid') / weights.sum()
    
    # HMA = WMA(2 * WMA(n/2) - WMA(n), sqrt(n))
    wma_half = wma(close_1w.values, half_len)
    wma_full = wma(close_1w.values, 21)
    raw_hma = 2 * wma_half - wma_full
    hma_21 = wma(raw_hma, sqrt_len)
    
    # Pad to match original length
    hma_21_padded = np.full(len(close_1w), np.nan)
    hma_21_padded[half_len - 1:] = hma_21
    hma_21_1w = hma_21_padded
    
    hma_21_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_21_1w)
    
    # Calculate 1d Donchian channels (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max()
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min()
    donchian_high = high_roll.values
    donchian_low = low_roll.values
    
    # Volume confirmation (2.0x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for calculations)
    start_idx = 60
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(hma_21_1w_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Donchian breakout long: close > upper band
            # Donchian breakout short: close < lower band
            breakout_long = close[i] > donchian_high[i]
            breakout_short = close[i] < donchian_low[i]
            
            # 1w HMA21 trend filter: close > HMA for longs, close < HMA for shorts
            hma_long = close[i] > hma_21_1w_aligned[i]
            hma_short = close[i] < hma_21_1w_aligned[i]
            
            if breakout_long and hma_long and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            elif breakout_short and hma_short and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Donchian breakdown (close < lower band) or trend reversal
            if close[i] < donchian_low[i] or close[i] < hma_21_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Donchian breakout (close > upper band) or trend reversal
            if close[i] > donchian_high[i] or close[i] > hma_21_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals