#!/usr/bin/env python3
"""
6h_ElderRay_ZeroLag_MA_Regime
Hypothesis: Elder Ray (Bull/Bear Power) combined with Zero Lag Moving Average crossover on 1d trend filter. Bull Power = High - EMA(13), Bear Power = EMA(13) - Low. Enter long when Bull Power > 0 and ZLMA crosses above signal line in uptrend (1d close > 1w EMA50). Enter short when Bear Power > 0 and ZLMA crosses below signal line in downtrend (1d close < 1w EMA50). Uses volume confirmation (>1.5x 20-bar MA) to reduce whipsaws. Designed for 6h timeframe to capture momentum with controlled frequency (target: 15-25 trades/year). Works in bull/bear markets by following 1d/1w trend while using Elder Ray for entry timing and ZLMA for reduced-lag signals.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d and 1w data ONCE before loop for HTF filters
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 50 or len(df_1w) < 20:
        return np.zeros(n)
    
    # 1d EMA13 for Elder Ray
    close_1d = df_1d['close'].values
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema_13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_13_1d)
    
    # 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = EMA13 - Low
    bull_power = high - ema_13_1d_aligned
    bear_power = ema_13_1d_aligned - low
    
    # Zero Lag Moving Average (ZLMA) on 6h close
    # ZLMA = EMA(close, Lag) where Lag = (period-1)/2
    period = 21
    lag = int((period - 1) / 2)
    ema1 = pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values
    ema2 = pd.Series(ema1).ewm(span=period, adjust=False, min_periods=period).mean().values
    zlma = 2 * ema1 - ema2  # Zero Lag MA
    
    # Signal line: EMA of ZLMA
    signal_line = pd.Series(zlma).ewm(span=9, adjust=False, min_periods=9).mean().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25  # Position size (25% of capital)
    
    # Warmup: max of calculations (21 for ZLMA, 13 for EMA13, 50 for 1w EMA, 20 for vol)
    start_idx = max(21, 13, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(zlma[i]) or np.isnan(signal_line[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        # Determine 1w trend: bullish if 1d close > 1w EMA50, bearish if 1d close < 1w EMA50
        # Use close price as proxy for 1d close in 6h timeframe (aligned)
        bullish_1w = close[i] > ema_50_1w_aligned[i]
        bearish_1w = close[i] < ema_50_1w_aligned[i]
        
        # ZLMA crossover signals
        zlma_cross_above = zlma[i] > signal_line[i] and zlma[i-1] <= signal_line[i-1]
        zlma_cross_below = zlma[i] < signal_line[i] and zlma[i-1] >= signal_line[i-1]
        
        # Entry conditions
        long_entry = bull_power[i] > 0 and zlma_cross_above and bullish_1w and volume_spike[i]
        short_entry = bear_power[i] > 0 and zlma_cross_below and bearish_1w and volume_spike[i]
        
        if position == 0:
            # Flat - look for entry
            if long_entry:
                signals[i] = base_size
                position = 1
            elif short_entry:
                signals[i] = -base_size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit on ZLMA cross below or trend change
            if zlma_cross_below or not bullish_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = base_size
        elif position == -1:
            # Short - exit on ZLMA cross above or trend change
            if zlma_cross_above or not bearish_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -base_size
    
    return signals

name = "6h_ElderRay_ZeroLag_MA_Regime"
timeframe = "6h"
leverage = 1.0