#!/usr/bin/env python3
"""
6h_ElderRay_ZeroLag_MA_Crossover_v2
Hypothesis: Elder Ray (Bull/Bear Power) with zero-lag moving average crossovers on 6h timeframe captures momentum shifts in both bull and bear markets. Uses 1d EMA for trend alignment and volume confirmation to filter false signals. Discrete sizing (0.25) balances return and fee drag. Target: 50-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA trend filter and volume average
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 1d volume average (20-period)
    vol_avg_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Zero-lag moving average (ZLMA) parameters
    zlma_period = 21
    # Calculate EMA
    ema = pd.Series(close).ewm(span=zlma_period, adjust=False, min_periods=zlma_period).mean().values
    # Calculate lag: EMA of EMA
    ema_of_ema = pd.Series(ema).ewm(span=zlma_period, adjust=False, min_periods=zlma_period).mean().values
    # ZLMA = 2*EMA - EMA_of_EMA
    zlma = 2 * ema - ema_of_ema
    
    # Elder Ray components: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Align all indicators to primary timeframe (6h)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    vol_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    zlma_aligned = align_htf_to_ltf(prices, df_1d, zlma)  # ZLMA is calculated from LTF close but aligned to ensure proper timing
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    size = 0.25   # Position size: 25% of capital (discrete level)
    
    # Warmup: need EMA34 (34), ZLMA (21*2), EMA13 (13), volume avg (20)
    start_idx = max(34, 42, 13, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_avg_1d_aligned[i]) or 
            np.isnan(zlma_aligned[i]) or np.isnan(bull_power_aligned[i]) or 
            np.isnan(bear_power_aligned[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema34 = ema34_1d_aligned[i]
        vol_avg = vol_avg_1d_aligned[i]
        zlma_val = zlma_aligned[i]
        bull_power_val = bull_power_aligned[i]
        bear_power_val = bear_power_aligned[i]
        volume_val = volume[i]
        
        if position == 0:
            # Determine trend alignment: price vs 1d EMA34
            uptrend = close_val > ema34
            downtrend = close_val < ema34
            
            # Volume confirmation: current volume > 1.5 * 1d volume average
            volume_confirm = volume_val > (1.5 * vol_avg)
            
            if uptrend and volume_confirm:
                # Long conditions: ZLMA crossover up AND Bull Power > 0
                if close_val > zlma_val and bull_power_val > 0:
                    signals[i] = size
                    position = 1
                    entry_price = close_val
            elif downtrend and volume_confirm:
                # Short conditions: ZLMA crossover down AND Bear Power < 0
                if close_val < zlma_val and bear_power_val < 0:
                    signals[i] = -size
                    position = -1
                    entry_price = close_val
        elif position == 1:
            # Exit conditions: ZLMA cross down OR Bear Power < 0
            if close_val < zlma_val or bear_power_val < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit conditions: ZLMA cross up OR Bull Power > 0
            if close_val > zlma_val or bull_power_val > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_ElderRay_ZeroLag_MA_Crossover_v2"
timeframe = "6h"
leverage = 1.0