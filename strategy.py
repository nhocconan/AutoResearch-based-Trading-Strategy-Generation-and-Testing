#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout + 1w HMA(21) trend filter + volume confirmation
# - Long: price > Donchian(20) high AND price > 1w HMA(21) AND volume > 1.5x 20-period average
# - Short: price < Donchian(20) low AND price < 1w HMA(21) AND volume > 1.5x 20-period average
# - Uses discrete position sizing (0.30) to minimize fee churn
# - ATR-based stoploss (2.0x ATR(14)) to manage risk
# - Designed for 1d timeframe: targets 7-25 trades/year to avoid fee drag
# - Works in bull/bear markets: 1w HMA filter prevents counter-trend trades, Donchian breakouts capture momentum

name = "1d_1w_donchian_hma_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Pre-compute 1w HMA(21) for trend filter
    close_1w = df_1w['close'].values
    hma_21 = calculate_hma(close_1w, 21)
    hma_21_aligned = align_htf_to_ltf(prices, df_1w, hma_21)
    
    # Pre-compute 1d Donchian(20) channels
    high_1d = prices['high'].values
    low_1d = prices['low'].values
    close_1d = prices['close'].values
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Pre-compute 1d volume confirmation
    volume_1d = prices['volume'].values
    avg_volume_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_1d > (1.5 * avg_volume_20)
    
    # Pre-compute 1d ATR(14) for stoploss
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(hma_21_aligned[i]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(vol_spike[i]) or np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price breaks below Donchian low OR stoploss hit
            if close_1d[i] < donchian_low[i] or close_1d[i] < entry_price - 2.0 * atr_14[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.30
                
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian high OR stoploss hit
            if close_1d[i] > donchian_high[i] or close_1d[i] > entry_price + 2.0 * atr_14[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.30
        else:  # Flat
            # Look for Donchian breakout with trend and volume filters
            if vol_spike[i]:
                # Long: price > Donchian high AND price > 1w HMA(21) (uptrend)
                if close_1d[i] > donchian_high[i] and close_1d[i] > hma_21_aligned[i]:
                    position = 1
                    entry_price = close_1d[i]
                    signals[i] = 0.30
                # Short: price < Donchian low AND price < 1w HMA(21) (downtrend)
                elif close_1d[i] < donchian_low[i] and close_1d[i] < hma_21_aligned[i]:
                    position = -1
                    entry_price = close_1d[i]
                    signals[i] = -0.30
    
    return signals

def calculate_hma(close, period):
    """Calculate Hull Moving Average"""
    if len(close) < period:
        return np.full_like(close, np.nan)
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    # WMA of half period
    wma_half = np.zeros_like(close)
    for i in range(half_period, len(close)):
        wma_half[i] = np.sum(close[i-half_period+1:i+1] * np.arange(1, half_period+1)) / (half_period * (half_period + 1) / 2)
    
    # WMA of full period
    wma_full = np.zeros_like(close)
    for i in range(period, len(close)):
        wma_full[i] = np.sum(close[i-period+1:i+1] * np.arange(1, period+1)) / (period * (period + 1) / 2)
    
    # Raw HMA = 2 * WMA(half) - WMA(full)
    raw_hma = 2 * wma_half - wma_full
    
    # Final HMA = WMA(sqrt_period) of raw_hma
    hma = np.zeros_like(close)
    for i in range(sqrt_period, len(close)):
        hma[i] = np.sum(raw_hma[i-sqrt_period+1:i+1] * np.arange(1, sqrt_period+1)) / (sqrt_period * (sqrt_period + 1) / 2)
    
    return hma