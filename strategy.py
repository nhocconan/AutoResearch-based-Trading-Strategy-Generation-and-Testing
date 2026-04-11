#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + HMA(21) trend + volume confirmation + ATR(14) stoploss
# - Long: price breaks above Donchian upper channel with HMA up and volume > 1.5x avg
# - Short: price breaks below Donchian lower channel with HMA down and volume > 1.5x avg
# - Exit: ATR-based trailing stop or opposite Donchian break
# - Uses 1d HMA for higher timeframe trend filter (bull: price > HMA, bear: price < HMA)
# - Designed to work in both bull and bear markets by following the trend with volatility stops
# - Target: 75-200 total trades over 4 years (19-50/year) to stay within fee drag limits

name = "4h_1d_donchian_hma_volume_v1"
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
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Load 1d data ONCE before loop for HMA trend filter (MTF rule compliance)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 21:
        return signals
    
    # Pre-compute 1d HMA(21)
    close_1d = df_1d['close'].values
    hma_21_1d = calculate_hma(close_1d, 21)
    hma_21_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_21_1d)
    
    # Pre-compute 4h Donchian channels (20-period)
    high_roll_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_upper = high_roll_max
    donchian_lower = low_roll_min
    
    # Pre-compute 4h ATR(14) for volatility and stoploss
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First TR is just high-low
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Pre-compute 4h volume SMA(20) for confirmation
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(hma_21_1d_aligned[i]) or np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or np.isnan(atr[i]) or 
            np.isnan(volume_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        high_price = high[i]
        low_price = low[i]
        volume_current = volume[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume_current > 1.5 * volume_sma_20[i]
        
        # Trend filter: 1d HMA direction
        hma_trend_up = close_price > hma_21_1d_aligned[i]
        hma_trend_down = close_price < hma_21_1d_aligned[i]
        
        # Donchian breakout conditions
        breakout_up = close_price > donchian_upper[i]
        breakout_down = close_price < donchian_lower[i]
        
        # ATR-based trailing stoploss (2.5 * ATR)
        atr_stop = 2.5 * atr[i]
        
        # Trading logic
        if position == 0:  # Flat - look for new entries
            # Long entry: bullish breakout with volume and trend alignment
            if breakout_up and vol_confirm and hma_trend_up:
                position = 1
                entry_price = close_price
                highest_since_entry = close_price
                signals[i] = 0.30
            
            # Short entry: bearish breakout with volume and trend alignment
            elif breakout_down and vol_confirm and hma_trend_down:
                position = -1
                entry_price = close_price
                lowest_since_entry = close_price
                signals[i] = -0.30
        
        elif position == 1:  # Long position - manage exit
            highest_since_entry = max(highest_since_entry, high_price)
            
            # Exit conditions: ATR trailing stop or Donchian breakdown
            trailing_stop = highest_since_entry - atr_stop
            donchian_exit = close_price < donchian_lower[i]
            
            if close_price <= trailing_stop or donchian_exit:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.30  # Maintain long
        
        elif position == -1:  # Short position - manage exit
            lowest_since_entry = min(lowest_since_entry, low_price)
            
            # Exit conditions: ATR trailing stop or Donchian breakout
            trailing_stop = lowest_since_entry + atr_stop
            donchian_exit = close_price > donchian_upper[i]
            
            if close_price >= trailing_stop or donchian_exit:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.30  # Maintain short
    
    return signals

def calculate_hma(close, period):
    """Calculate Hull Moving Average"""
    if len(close) < period:
        return np.full_like(close, np.nan)
    
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    # WMA of half period
    wma_half = pd.Series(close).ewm(span=half_period, adjust=False, min_periods=half_period).mean().values
    
    # WMA of full period
    wma_full = pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    # Raw HMA: 2*WMA(half) - WMA(full)
    raw_hma = 2 * wma_half - wma_full
    
    # Final HMA: WMA of raw_hma with sqrt_period
    hma = pd.Series(raw_hma).ewm(span=sqrt_period, adjust=False, min_periods=sqrt_period).mean().values
    
    return hma