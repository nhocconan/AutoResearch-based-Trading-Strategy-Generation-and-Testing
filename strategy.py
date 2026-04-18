#!/usr/bin/env python3
"""
4h_Adaptive_Kelly_Adjusted_Donchian_Breakout_Volume
Hypothesis: Donchian(20) breakout with volume confirmation and Kelly-adjusted sizing.
Adaptive position sizing based on volatility and signal strength reduces drawdown in bear markets.
Works in bull (breakouts) and bear (mean-reversion at bands) via volatility-adjusted exits.
Target: 20-30 trades/year for low fee drag and high edge.
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
    
    # Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # ATR for volatility and stop
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.inf], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume spike: >1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Kelly fraction estimate: (win_rate * avg_win - avg_loss) / avg_win
    # Simplified: use volatility-adjusted signal strength
    price_change = np.abs(close - np.roll(close, 1))
    price_change[0] = 0
    vol_adj_strength = pd.Series(price_change).rolling(window=10, min_periods=10).mean().values / (atr + 1e-8)
    kelly_fraction = np.clip(vol_adj_strength * 0.5, 0.1, 0.4)  # cap at 0.4, min 0.1
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 20  # Warmup for Donchian
    
    for i in range(start_idx, n):
        if (np.isnan(high_20[i]) or 
            np.isnan(low_20[i]) or
            np.isnan(atr[i]) or
            np.isnan(volume_spike[i]) or
            np.isnan(kelly_fraction[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper = high_20[i]
        lower = low_20[i]
        vol_spike = volume_spike[i]
        kelly = kelly_fraction[i]
        
        if position == 0:
            # Long: break above upper band with volume
            if price > upper and vol_spike:
                signals[i] = kelly
                position = 1
            # Short: break below lower band with volume
            elif price < lower and vol_spike:
                signals[i] = -kelly
                position = -1
        
        elif position == 1:
            signals[i] = kelly
            # Exit: price closes below lower band OR ATR-based stop
            if price < lower:
                signals[i] = 0.0
                position = 0
            elif price < high[i] - 2.0 * atr[i]:  # trailing stop
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -kelly
            # Exit: price closes above upper band OR ATR-based stop
            if price > upper:
                signals[i] = 0.0
                position = 0
            elif price > low[i] + 2.0 * atr[i]:  # trailing stop
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Adaptive_Kelly_Adjusted_Donchian_Breakout_Volume"
timeframe = "4h"
leverage = 1.0