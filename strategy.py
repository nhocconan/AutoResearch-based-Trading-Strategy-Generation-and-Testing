#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using 1w Donchian breakout with volume confirmation and ATR volatility filter
# - Uses weekly Donchian channels (20-period) to identify structural breaks
# - Requires volume > 1.5x 20-period average for confirmation
# - Filters out low volatility environments using ATR ratio (current ATR < 0.5 * 20-period ATR average)
# - Exits when price crosses opposite Donchian boundary or volatility spikes (ATR ratio > 2.0)
# - Designed to capture strong trending moves while avoiding choppy markets
# - Target: 30-80 total trades over 4 years (7-20/year) with 0.25 position sizing

name = "1d_WeeklyDonchian_Breakout_Volume_ATRFilter"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for Donchian channel calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 1w Donchian channels (20-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Upper band: highest high over 20 periods
    donchian_high = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    # Lower band: lowest low over 20 periods
    donchian_low = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Calculate ATR (14) for volatility filtering
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma = pd.Series(atr).rolling(window=20, min_periods=20).mean().values
    
    # Align 1w indicators to 1d timeframe
    donchian_high_1d = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_1d = align_htf_to_ltf(prices, df_1w, donchian_low)
    atr_ma_1d = align_htf_to_ltf(prices, df_1w, atr_ma)
    
    # Volume filters (1d timeframe)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)  # Volume confirmation
    
    # ATR ratio for volatility regime filter
    atr_14 = pd.Series(
        np.maximum(
            np.maximum(high - low, 
                      np.abs(high - np.roll(close, 1))),
            np.abs(low - np.roll(close, 1))
        )
    ).rolling(window=14, min_periods=14).mean().values
    atr_ratio = atr_14 / (atr_ma_1d + 1e-10)  # Current ATR vs 20-period average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup
        # Skip if any critical value is NaN
        if (np.isnan(donchian_high_1d[i]) or np.isnan(donchian_low_1d[i]) or 
            np.isnan(atr_ma_1d[i]) or np.isnan(volume_spike[i]) or
            np.isnan(atr_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Look for breakout with volume confirmation and low volatility regime
            # Low volatility: ATR ratio < 0.5 (avoid choppy markets)
            low_vol = atr_ratio[i] < 0.5
            
            if low_vol and volume_spike[i]:
                # Long: price breaks above weekly Donchian high
                if close[i] > donchian_high_1d[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: price breaks below weekly Donchian low
                elif close[i] < donchian_low_1d[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: price crosses below weekly Donchian low OR volatility spikes (ATR ratio > 2.0)
            if close[i] < donchian_low_1d[i] or atr_ratio[i] > 2.0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above weekly Donchian high OR volatility spikes (ATR ratio > 2.0)
            if close[i] > donchian_high_1d[i] or atr_ratio[i] > 2.0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals