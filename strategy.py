#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using weekly Donchian channel breakouts with daily volume confirmation and volatility filter.
# Long when price breaks above weekly Donchian high with volume above average.
# Short when price breaks below weekly Donchian low with volume above average.
# Uses ATR-based volatility filter to avoid choppy markets.
# Designed for low trade frequency (10-25/year) to minimize fee impact and capture strong trends.

name = "1d_WeeklyDonchian_Breakout_Volume_Volatility"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Donchian channel calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate weekly Donchian channels (20-period)
    donch_high = np.full_like(high_1w, np.nan)
    donch_low = np.full_like(low_1w, np.nan)
    
    for i in range(20, len(high_1w)):
        donch_high[i] = np.max(high_1w[i-20:i])
        donch_low[i] = np.min(low_1w[i-20:i])
    
    # Align Donchian channels to daily timeframe
    donch_high_aligned = align_htf_to_ltf(prices, df_1w, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_1w, donch_low)
    
    # Daily ATR(14) for volatility filter
    high_low = high - low
    high_close = np.abs(high - np.roll(close, 1))
    low_close = np.abs(low - np.roll(close, 1))
    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    true_range[0] = high_low[0]  # First value
    
    atr = np.zeros_like(close)
    for i in range(14, len(true_range)):
        atr[i] = np.mean(true_range[i-14:i])
    
    # Daily volume confirmation: volume > 1.5x 20-period EMA
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm = volume > (vol_ema * 1.5)
    
    # Volatility filter: ATR > 0.5x 50-period SMA of ATR (avoid low volatility/chop)
    atr_sma = pd.Series(atr).rolling(window=50, min_periods=50).mean().values
    vol_filter = atr > (atr_sma * 0.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Ensure enough data for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or
            np.isnan(atr[i]) or np.isnan(atr_sma[i]) or np.isnan(vol_ema[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: break above weekly Donchian high with volume and volatility
            if (close[i] > donch_high_aligned[i] and 
                vol_confirm[i] and 
                vol_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: break below weekly Donchian low with volume and volatility
            elif (close[i] < donch_low_aligned[i] and 
                  vol_confirm[i] and 
                  vol_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: break below weekly Donchian low or volatility drops
            if (close[i] < donch_low_aligned[i] or not vol_filter[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: break above weekly Donchian high or volatility drops
            if (close[i] > donch_high_aligned[i] or not vol_filter[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals