#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_VDC_Breakout_1dTrend_Confirm"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once for volatility and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1d volatility: ATR(14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([np.array([np.nan]), tr])
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 1d trend: EMA50
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_1d = close_1d > ema50_1d
    
    # Align to 12h
    atr14_12h = align_htf_to_ltf(prices, df_1d, atr14)
    trend_1d_12h = align_htf_to_ltf(prices, df_1d, trend_1d.astype(float))
    
    # Volatility contraction: current ATR < 0.6 * 20-period ATR average
    atr_ma20 = pd.Series(atr14).rolling(window=20, min_periods=20).mean().values
    atr_ma20_12h = align_htf_to_ltf(prices, df_1d, atr_ma20)
    vol_contract = atr14 < (atr_ma20 * 0.6)
    vol_contract_12h = align_htf_to_ltf(prices, df_1d, vol_contract.astype(float))
    
    # 12h price channel: Donchian(20) breakout
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # warmup for ATR and Donchian
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(atr14_12h[i]) or np.isnan(trend_1d_12h[i]) or 
            np.isnan(vol_contract_12h[i]) or np.isnan(donch_high[i]) or np.isnan(donch_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: volatility contraction + price breaks above Donchian high + 1d uptrend
            long_cond = (vol_contract_12h[i] > 0.5 and 
                        close[i] > donch_high[i] and 
                        trend_1d_12h[i] > 0.5)
            
            # Short: volatility contraction + price breaks below Donchian low + 1d downtrend
            short_cond = (vol_contract_12h[i] > 0.5 and 
                         close[i] < donch_low[i] and 
                        trend_1d_12h[i] < 0.5)
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price closes below Donchian low or volatility expansion
            if close[i] < donch_low[i] or vol_contract_12h[i] < 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above Donchian high or volatility expansion
            if close[i] > donch_high[i] or vol_contract_12h[i] < 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Volatility contraction (ATR < 60% of 20-period average) precedes breakouts.
# Uses 12h timeframe with 1d ATR and trend filters. Enter on Donchian(20) breakout in direction of 1d trend.
# Exit when price reverses to opposite Donchian band or volatility expands.
# Target: 20-40 trades/year to avoid fee drag while capturing explosive moves in both bull and bear markets.
# Volatility contraction is a proven precursor to breakouts, effective in ranging and trending markets.