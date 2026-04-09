#!/usr/bin/env python3
# 1d_volatility_breakout_volume_v1
# Hypothesis: 1d strategy combining ATR-based volatility breakout with volume confirmation
# and weekly trend filter. In ranging/bear markets (2025+), volatility expansions from
# weekly pivot areas with volume confirmation capture meaningful moves while avoiding
# false breakouts. Uses discrete sizing (0.0, ±0.30) to limit fee churn and target
# 10-20 trades/year for sustainability.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_volatility_breakout_volume_v1"
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
    
    # 1w HTF data for weekly trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Weekly EMA(21) for trend filter
    close_w = pd.Series(df_1w['close'].values)
    ema_21_w = close_w.ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_w)
    
    # Daily ATR(14) for volatility breakout
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Daily Donchian(20) breakout levels
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(ema_21_w_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        if position == 1:  # Long position
            # Exit: price closes below weekly EMA(21) or volatility contraction
            if close[i] < ema_21_w_aligned[i] or volume[i] < 0.8 * volume_ma[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.30
                
        elif position == -1:  # Short position
            # Exit: price closes above weekly EMA(21) or volatility contraction
            if close[i] > ema_21_w_aligned[i] or volume[i] < 0.8 * volume_ma[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.30
        else:  # Flat
            if volume_confirmed:
                # Long entry: price breaks above Donchian(20) high with volume
                if high[i] > highest_high[i]:
                    position = 1
                    signals[i] = 0.30
                # Short entry: price breaks below Donchian(20) low with volume
                elif low[i] < lowest_low[i]:
                    position = -1
                    signals[i] = -0.30
    
    return signals