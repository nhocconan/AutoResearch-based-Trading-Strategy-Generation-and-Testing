#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Bollinger Band squeeze breakout with 1d ATR regime filter
    # Bollinger Band squeeze (low volatility) precedes explosive moves in both bull/bear markets
    # Only trade breakouts when 1d ATR is elevated (trending regime) to avoid whipsaws in chop
    # Uses discrete sizing 0.25 to minimize fee churn. Target: 15-25 trades/year per symbol.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for ATR regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d ATR(14) for regime filter
    tr_1d = np.maximum(
        high_1d[1:] - low_1d[1:],
        np.maximum(
            np.abs(high_1d[1:] - close_1d[:-1]),
            np.abs(low_1d[1:] - close_1d[:-1])
        )
    )
    tr_1d = np.concatenate([[np.nan], tr_1d])  # align length
    atr_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate Bollinger Bands (20, 2.0) on 6h
    bb_period = 20
    bb_std = 2.0
    bb_ma = np.full(n, np.nan)
    bb_std_dev = np.full(n, np.nan)
    bb_upper = np.full(n, np.nan)
    bb_lower = np.full(n, np.nan)
    
    for i in range(bb_period, n):
        bb_ma[i] = np.mean(close[i-bb_period:i])
        bb_std_dev[i] = np.std(close[i-bb_period:i])
        bb_upper[i] = bb_ma[i] + (bb_std * bb_std_dev[i])
        bb_lower[i] = bb_ma[i] - (bb_std * bb_std_dev[i])
    
    # Bollinger Band Width (normalized)
    bb_width = np.full(n, np.nan)
    bb_width = (bb_upper - bb_lower) / bb_ma
    
    # Bollinger Band Squeeze: BB Width < 20th percentile of last 50 periods
    bb_width_percentile = np.full(n, np.nan)
    for i in range(50, n):
        bb_width_percentile[i] = np.percentile(bb_width[i-50:i], 20)
    
    squeeze = bb_width < bb_width_percentile
    
    # Breakout detection: price breaks above/below Bollinger Bands
    breakout_up = close > bb_upper
    breakout_down = close < bb_lower
    
    # Volume confirmation (optional but helpful): volume > 1.5 * 20-period average
    volume = prices['volume'].values
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(atr_1d_aligned[i]) or np.isnan(bb_ma[i]) or 
            np.isnan(bb_width_percentile[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: only trade when 1d ATR is above its 50-period median (trending regime)
        atr_median = np.full(n, np.nan)
        if i >= 100:  # need sufficient history for median
            atr_median[i] = np.nanmedian(atr_1d_aligned[i-50:i])
        else:
            atr_median[i] = np.nan
        
        trending_regime = ~np.isnan(atr_median) and (atr_1d_aligned[i] > atr_median[i])
        
        # Entry logic: Bollinger Band breakout during squeeze + trending regime
        long_entry = False
        short_entry = False
        
        if trending_regime and squeeze[i]:
            long_entry = breakout_up[i] and volume_spike[i]
            short_entry = breakout_down[i] and volume_spike[i]
        
        # Exit logic: return to Bollinger Band middle or opposite breakout
        long_exit = (close[i] < bb_ma[i]) or breakout_down[i]
        short_exit = (close[i] > bb_ma[i]) or breakout_up[i]
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_1d_bb_squeeze_breakout_atr_regime_v1"
timeframe = "6h"
leverage = 1.0