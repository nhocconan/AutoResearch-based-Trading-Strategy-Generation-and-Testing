#!/usr/bin/env python3
"""
6h_Keltner_Breakout_Volume_Regime_v1
Hypothesis: On 6h timeframe, enter long when price breaks above Keltner upper band (EMA20 + 2*ATR10) AND volume > 1.5x 20-period average volume AND ADX(14) > 25 (trending regime). Enter short when price breaks below Keltner lower band (EMA20 - 2*ATR10) AND volume spike AND ADX > 25. Uses discrete sizing (0.0, ±0.25) to limit fee drag. Keltner channels adapt to volatility, reducing false breakouts in choppy markets. Volume confirmation ensures participation. ADX filter ensures we only trade in trending regimes, avoiding whipsaws in ranges. Designed to generate ~15-25 trades per year on BTC/ETH/SOL with Sharpe > 0 in both bull and bear regimes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate EMA20 for Keltner middle band
    close_s = pd.Series(close)
    ema_20 = close_s.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate ATR(10) for Keltner bands
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_10 = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Keltner bands
    keltner_upper = ema_20 + 2.0 * atr_10
    keltner_lower = ema_20 - 2.0 * atr_10
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 1.5 * volume_ma
    
    # ADX(14) for trend regime filter
    # +DM and -DM
    up_move = high[1:] - high[:-1]
    down_move = low[:-1] - low[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Smoothed +DM, -DM, TR
    tr_rma = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    plus_dm_rma = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    minus_dm_rma = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # +DI and -DI
    plus_di = 100 * plus_dm_rma / tr_rma
    minus_di = 100 * minus_dm_rma / tr_rma
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Trend regime: ADX > 25
    trending_regime = adx > 25.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need EMA20, ATR10, volume MA, ADX warmup
    start_idx = max(20, 10, 20, 14*2)  # ~28 bars
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(keltner_upper[i]) or 
            np.isnan(keltner_lower[i]) or 
            np.isnan(volume_ma[i]) or 
            np.isnan(adx[i]) or 
            np.isnan(trending_regime[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Breakout conditions
        breakout_up = close[i] > keltner_upper[i]
        breakout_down = close[i] < keltner_lower[i]
        
        if position == 0:
            # Long: breakout above upper band + volume spike + trending regime
            long_signal = breakout_up and volume_spike[i] and trending_regime[i]
            
            # Short: breakout below lower band + volume spike + trending regime
            short_signal = breakout_down and volume_spike[i] and trending_regime[i]
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price falls below middle band OR ADX drops below 20 (regime change)
            if close[i] < ema_20[i] or adx[i] < 20.0:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price rises above middle band OR ADX drops below 20 (regime change)
            if close[i] > ema_20[i] or adx[i] < 20.0:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Keltner_Breakout_Volume_Regime_v1"
timeframe = "6h"
leverage = 1.0