#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h HMA21 trend filter and volume confirmation.
# Long when price breaks above 4h Donchian upper band AND 12h close > HMA21 (bullish trend) AND volume > 2.0x 20-bar average.
# Short when price breaks below 4h Donchian lower band AND 12h close < HMA21 (bearish trend) AND volume > 2.0x 20-bar average.
# Uses discrete sizing 0.25 to manage drawdown. Target: 75-200 total trades over 4 years (19-50/year).
# Donchian channels provide robust price channels, HMA21 filters trend with low lag, volume spike confirms momentum.
# Primary timeframe: 4h, HTF: 12h for HMA trend filter.

name = "4h_Donchian20_12hHMA21_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop for HMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 21:
        return np.zeros(n)
    
    # Calculate 4h Donchian channels (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_upper = high_roll
    donchian_lower = low_roll
    
    # Calculate 12h HMA(21) trend filter
    def hull_moving_average(arr, period):
        """Calculate Hull Moving Average"""
        half_period = period // 2
        sqrt_period = int(np.sqrt(period))
        
        # WMA for half period
        weights_half = np.arange(1, half_period + 1)
        wma_half = np.convolve(arr, weights_half, mode='full')[-len(arr):] / weights_half.sum()
        wma_half[:half_period-1] = np.nan
        
        # WMA for full period
        weights_full = np.arange(1, period + 1)
        wma_full = np.convolve(arr, weights_full, mode='full')[-len(arr):] / weights_full.sum()
        wma_full[:period-1] = np.nan
        
        # Raw HMA: 2*WMA(half) - WMA(full)
        raw_hma = 2 * wma_half - wma_full
        
        # Final WMA of raw HMA with sqrt period
        weights_sqrt = np.arange(1, sqrt_period + 1)
        wma_sqrt = np.convolve(raw_hma, weights_sqrt, mode='full')[-len(raw_hma):] / weights_sqrt.sum()
        wma_sqrt[:sqrt_period-1] = np.nan
        
        return wma_sqrt
    
    close_12h = df_12h['close'].values
    hma_21_12h = hull_moving_average(close_12h, 21)
    hma_21_aligned = align_htf_to_ltf(prices, df_12h, hma_21_12h)
    
    # Volume confirmation: current 4h volume > 2.0x 20-bar average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for Donchian and HMA
    
    for i in range(start_idx, n):
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or \
           np.isnan(hma_21_aligned[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_vol = volume[i]
        curr_vol_ma = vol_ma[i]
        
        if curr_vol_ma <= 0:
            signals[i] = 0.0
            continue
            
        volume_confirm = curr_vol > (curr_vol_ma * 2.0)  # Volume spike threshold
        
        # Donchian breakout signals
        breakout_up = curr_high > donchian_upper[i]  # break above upper band
        breakout_down = curr_low < donchian_lower[i]  # break below lower band
        
        # Trend filter: bullish if close > HMA21, bearish if close < HMA21
        bullish_trend = curr_close > hma_21_aligned[i]
        bearish_trend = curr_close < hma_21_aligned[i]
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: breakout above upper band AND bullish trend AND volume confirmation
            if (breakout_up and 
                bullish_trend and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: breakout below lower band AND bearish trend AND volume confirmation
            elif (breakout_down and 
                  bearish_trend and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price crosses below lower band (stoploss) OR trend turns bearish
            if (curr_low < donchian_lower[i] or 
                bearish_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price crosses above upper band (stoploss) OR trend turns bullish
            if (curr_high > donchian_upper[i] or 
                bullish_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals