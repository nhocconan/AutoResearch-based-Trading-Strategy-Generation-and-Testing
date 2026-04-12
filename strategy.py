#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h_1d_vortex_trend_v1
# Uses Vortex indicator from 1-day chart to determine trend direction on 4h timeframe.
# In bull markets, VI+ > VI- indicates uptrend; in bear markets, VI- > VI+ indicates downtrend.
# Entry when Vortex crossover aligns with price above/below 20-period EMA for confirmation.
# Includes volume confirmation to filter low-quality signals and ATR-based stoploss via signal=0.
# Target: 20-40 trades/year per symbol for low friction and high win rate.
name = "4h_1d_vortex_trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Vortex calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate True Range for Vortex
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Calculate Vortex Indicator components
    vm_plus = np.abs(high_1d[1:] - low_1d[:-1])
    vm_minus = np.abs(low_1d[1:] - high_1d[:-1])
    
    # Sum over 14 periods
    period = 14
    tr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    vm_plus_sum = pd.Series(vm_plus).rolling(window=period, min_periods=period).sum().values
    vm_minus_sum = pd.Series(vm_minus).rolling(window=period, min_periods=period).sum().values
    
    # Avoid division by zero
    vi_plus = np.where(tr_sum != 0, vm_plus_sum / tr_sum, 0)
    vi_minus = np.where(tr_sum != 0, vm_minus_sum / tr_sum, 0)
    
    # Align Vortex to 4h timeframe
    vi_plus_aligned = align_htf_to_ltf(prices, df_1d, vi_plus)
    vi_minus_aligned = align_htf_to_ltf(prices, df_1d, vi_minus)
    
    # Determine trend: VI+ > VI- = uptrend, VI- > VI+ = downtrend
    trend_up = vi_plus_aligned > vi_minus_aligned
    trend_down = vi_minus_aligned > vi_plus_aligned
    
    # EMA confirmation on 4h
    ema_period = 20
    ema = pd.Series(close).ewm(span=ema_period, adjust=False, min_periods=ema_period).mean().values
    price_above_ema = close > ema
    price_below_ema = close < ema
    
    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # start after warmup
        # Skip if Vortex not ready
        if np.isnan(vi_plus_aligned[i]) or np.isnan(vi_minus_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Check volume filter
        if not vol_confirm[i]:
            # Hold current position if volume filter fails
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Long signal: uptrend + price above EMA
        if trend_up[i] and price_above_ema[i] and position != 1:
            position = 1
            signals[i] = 0.25
        # Short signal: downtrend + price below EMA
        elif trend_down[i] and price_below_ema[i] and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit conditions: trend reversal
        elif trend_down[i] and position == 1:
            position = 0
            signals[i] = 0.0
        elif trend_up[i] and position == -1:
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