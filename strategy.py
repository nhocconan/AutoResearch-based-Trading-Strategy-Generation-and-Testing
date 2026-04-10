#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h TRIX + volume spike + choppiness regime filter
# - TRIX(15) crossing zero line as momentum signal
# - Volume > 1.5x 20-period average for confirmation
# - Choppiness Index(14) > 61.8 for ranging market (mean reversion) or < 38.2 for trending
# - In ranging markets (CHOP > 61.8): fade TRIX crosses (sell on bullish cross, buy on bearish cross)
# - In trending markets (CHOP < 38.2): follow TRIX crosses (buy on bullish cross, sell on bearish cross)
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Targets 12-37 trades/year (50-150 total over 4 years) to avoid fee drag
# - Works in both bull and bear markets by adapting to regime

name = "12h_1d_trix_volume_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 1d indicators
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # TRIX(15): triple exponential moving average
    # TRIX = EMA(EMA(EMA(close, 15), 15), 15)
    ema1 = pd.Series(close_1d).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    trix = pd.Series(ema3).pct_change() * 100  # Percentage change
    trix_values = trix.values
    
    # Align TRIX to 12h timeframe
    trix_aligned = align_htf_to_ltf(prices, df_1d, trix_values)
    
    # Volume confirmation: > 1.5x 20-period average
    avg_volume_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_1d > (1.5 * avg_volume_20)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike)
    
    # Choppiness Index(14)
    # CHOP = 100 * log10(sum(ATR(14)) / (log10(n) * (max(high,n) - min(low,n))))
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period has no previous close
    
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    max_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    range_14 = max_high_14 - min_low_14
    range_14 = np.where(range_14 == 0, 1e-10, range_14)
    
    chop = 100 * np.log10(atr_14 / (np.log10(14) * range_14))
    chop_values = chop
    
    # Align Choppiness Index to 12h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_values)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    atr_stop_multiplier = 2.5
    
    # Calculate ATR for stoploss (using 1d data)
    high_low = df_1d['high'] - df_1d['low']
    high_close = np.abs(df_1d['high'] - df_1d['close'].shift(1))
    low_close = np.abs(df_1d['low'] - df_1d['close'].shift(1))
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_ranges = np.nanmax(ranges.values, axis=1)
    atr_14_1d = pd.Series(true_ranges).rolling(window=14, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(trix_aligned[i]) or np.isnan(vol_spike_aligned[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(atr_14_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # ATR-based stoploss
            if prices['close'].iloc[i] < entry_price - atr_stop_multiplier * atr_14_1d_aligned[i]:
                position = 0
                entry_price = 0.0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # ATR-based stoploss
            if prices['close'].iloc[i] > entry_price + atr_stop_multiplier * atr_14_1d_aligned[i]:
                position = 0
                entry_price = 0.0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # TRIX zero line cross signals
            trix_now = trix_aligned[i]
            trix_prev = trix_aligned[i-1] if i > 0 else 0
            
            bullish_cross = trix_prev <= 0 and trix_now > 0
            bearish_cross = trix_prev >= 0 and trix_now < 0
            
            # Regime-based logic
            chop_now = chop_aligned[i]
            is_ranging = chop_now > 61.8  # Choppy/ranging market
            is_trending = chop_now < 38.2  # Trending market
            
            # Volume confirmation
            vol_confirmed = vol_spike_aligned[i]
            
            # In ranging markets: mean reversion (fade the move)
            # In trending markets: trend following
            if is_ranging and vol_confirmed:
                # Fade TRIX crosses in ranging markets
                if bullish_cross:
                    # Sell on bullish TRIX cross (expecting mean reversion down)
                    position = -1
                    entry_price = prices['open'].iloc[i+1] if i+1 < n else prices['close'].iloc[i]
                    signals[i] = -0.25
                elif bearish_cross:
                    # Buy on bearish TRIX cross (expecting mean reversion up)
                    position = 1
                    entry_price = prices['open'].iloc[i+1] if i+1 < n else prices['close'].iloc[i]
                    signals[i] = 0.25
                    
            elif is_trending and vol_confirmed:
                # Follow TRIX crosses in trending markets
                if bullish_cross:
                    # Buy on bullish TRIX cross
                    position = 1
                    entry_price = prices['open'].iloc[i+1] if i+1 < n else prices['close'].iloc[i]
                    signals[i] = 0.25
                elif bearish_cross:
                    # Sell on bearish TRIX cross
                    position = -1
                    entry_price = prices['open'].iloc[i+1] if i+1 < n else prices['close'].iloc[i]
                    signals[i] = -0.25
    
    return signals