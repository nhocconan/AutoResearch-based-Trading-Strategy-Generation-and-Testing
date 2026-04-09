#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot breakout with 4h volume confirmation and 1d chop regime filter
# Camarilla pivots provide precise intraday support/resistance levels
# 4h volume spike confirms breakout authenticity
# 1d chop regime filter adapts to market conditions: trend follow in trending markets, mean revert in ranging
# Session filter (08-20 UTC) reduces noise during low-liquidity hours
# Target: 60-150 total trades over 4 years (15-37/year) with discrete sizing 0.20

name = "1h_4h_1d_camarilla_volume_chop_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 4h data ONCE for volume confirmation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Calculate 4h average volume (20-period)
    volume_4h = df_4h['volume'].values
    volume_s_4h = pd.Series(volume_4h)
    avg_volume_4h = volume_s_4h.rolling(window=20, min_periods=20).mean().values
    avg_volume_4h_aligned = align_htf_to_ltf(prices, df_4h, avg_volume_4h)
    
    # Load 1d data ONCE for chop regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d Choppiness Index (CHOP)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Wilder's smoothing for ATR
    def wilders_smoothing(values, period):
        if len(values) < period:
            return np.full(len(values), np.nan)
        alpha = 1.0 / period
        result = np.full(len(values), np.nan)
        result[period-1] = np.nanmean(values[:period])
        for i in range(period, len(values)):
            result[i] = alpha * values[i] + (1 - alpha) * result[i-1]
        return result
    
    atr_1d = wilders_smoothing(tr, 14)
    
    # Highest high and lowest low over 14 periods
    hh_1d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_1d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Chop calculation
    sum_atr_14 = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    range_14 = hh_1d - ll_1d
    chop_1d = np.where(range_14 != 0, 
                       100 * np.log10(sum_atr_14 / range_14) / np.log10(14), 
                       50)
    
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if outside session or missing data
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        if (np.isnan(avg_volume_4h_aligned[i]) or np.isnan(chop_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Calculate Camarilla pivots for 1h timeframe using previous bar's OHLC
        if i < 1:
            signals[i] = 0.0
            continue
            
        prev_close = close[i-1]
        prev_high = high[i-1]
        prev_low = low[i-1]
        prev_range = prev_high - prev_low
        
        # Camarilla levels
        h5 = prev_close + 1.1 * prev_range / 2  # Resistance 5
        h4 = prev_close + 1.1 * prev_range / 4  # Resistance 4
        h3 = prev_close + 1.1 * prev_range / 6  # Resistance 3
        l3 = prev_close - 1.1 * prev_range / 6  # Support 3
        l4 = prev_close - 1.1 * prev_range / 4  # Support 4
        l5 = prev_close - 1.1 * prev_range / 2  # Support 5
        
        # Volume confirmation: current 1h volume > 1.5x 4h average volume
        volume_confirmed = volume[i] > 1.5 * avg_volume_4h_aligned[i]
        
        # Regime filters
        trending_regime = chop_1d_aligned[i] < 38.2
        ranging_regime = chop_1d_aligned[i] > 61.8
        
        if position == 1:  # Long position
            # Exit: price closes below H3 or regime shifts to ranging
            if close[i] < h3 or ranging_regime:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: price closes above L3 or regime shifts to ranging
            if close[i] > l3 or ranging_regime:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Entry logic
            if trending_regime:
                # Follow breakout in trending regime
                if close[i] > h5 and volume_confirmed:
                    position = 1
                    signals[i] = 0.20
                elif close[i] < l5 and volume_confirmed:
                    position = -1
                    signals[i] = -0.20
            elif ranging_regime:
                # Mean revert at H3/L3 in ranging regime
                if close[i] < l3 and volume_confirmed:
                    position = 1
                    signals[i] = 0.20
                elif close[i] > h3 and volume_confirmed:
                    position = -1
                    signals[i] = -0.20
    
    return signals