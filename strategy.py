#!/usr/bin/env python3
"""
6h RSI(9) + 1d Bollinger Bands Width + Volume Spike
Hypothesis: In low volatility (BB width low), RSI extremes signal reversals.
Volume spike confirms conviction. Works in bull/bear as mean reversion in ranging markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_rsi_bb_width_volume_spike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === RSI(9) on 6h ===
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/9, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/9, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # === 1d Bollinger Bands Width (20, 2) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    sma_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper = sma_20 + 2 * std_20
    lower = sma_20 - 2 * std_20
    bb_width = (upper - lower) / (sma_20 + 1e-10)
    bb_width_avg = pd.Series(bb_width).rolling(window=50, min_periods=50).mean().values
    bb_width_aligned = align_htf_to_ltf(prices, df_1d, bb_width)
    bb_width_avg_aligned = align_htf_to_ltf(prices, df_1d, bb_width_avg)
    
    # === Volume Spike (20-bar avg) ===
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume / (vol_avg + 1e-10)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        if np.isnan(rsi[i]) or np.isnan(bb_width_aligned[i]) or np.isnan(bb_width_avg_aligned[i]) or np.isnan(vol_spike[i]):
            signals[i] = 0.0
            continue
        
        # Low volatility regime: BB width below average
        low_vol = bb_width_aligned[i] < bb_width_avg_aligned[i]
        
        if position == 1:  # Long position
            # Exit: RSI crosses above 60 (overbought)
            if rsi[i] > 60:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: RSI crosses below 40 (oversold)
            if rsi[i] < 40:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Mean reversion in low volatility with volume confirmation
            if low_vol and vol_spike[i] > 1.5:
                if rsi[i] < 30:  # Oversold
                    position = 1
                    signals[i] = 0.25
                elif rsi[i] > 70:  # Overbought
                    position = -1
                    signals[i] = -0.25
    
    return signals