#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1-day ATR-based volatility breakout with daily mean-reversion filter.
# Uses ATR(14) expansion to capture volatility bursts after low volatility periods.
# Mean-reversion filter: RSI(14) < 40 for longs, > 60 for shorts to avoid chasing momentum.
# Volume confirmation: > 1.3x 20-period average to validate breakout strength.
# Designed for 12h timeframe to capture multi-day moves while minimizing trade frequency.
# Works in bull markets by catching breakout continuations and in bear markets by fading overextended moves.
# Target: 15-30 trades/year per symbol (60-120 total over 4 years) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE for ATR and RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need sufficient data for ATR and RSI
        return np.zeros(n)
    
    # Calculate ATR(14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period TR is just high-low
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate RSI(14)
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_14 = 100 - (100 / (1 + rs))
    
    # Align ATR and RSI to 12h timeframe
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    rsi_14_aligned = align_htf_to_ltf(prices, df_1d, rsi_14)
    
    # Calculate ATR ratio: current ATR / 20-period average ATR
    atr_ma_20 = pd.Series(atr_14_aligned).rolling(window=20, min_periods=20).mean().values
    atr_ratio = atr_14_aligned / atr_ma_20
    
    # Volume confirmation: 1.3x average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(30, 20)  # Need ATR, RSI and volume MA
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(atr_ratio[i]) or 
            np.isnan(rsi_14_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volatility breakout condition: ATR ratio > 1.5
        vol_breakout = atr_ratio[i] > 1.5
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.3 * vol_ma[i]
        
        if position == 0:
            # Look for volatility breakout with mean-reversion filter
            if vol_breakout and volume_confirmed:
                # Long: volatility expansion + RSI < 40 (oversold)
                if rsi_14_aligned[i] < 40:
                    position = 1
                    signals[i] = position_size
                # Short: volatility expansion + RSI > 60 (overbought)
                elif rsi_14_aligned[i] > 60:
                    position = -1
                    signals[i] = -position_size
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: volatility contraction or RSI > 70 (overbought)
            if (atr_ratio[i] < 1.0 or 
                rsi_14_aligned[i] > 70):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: volatility contraction or RSI < 30 (oversold)
            if (atr_ratio[i] < 1.0 or 
                rsi_14_aligned[i] < 30):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1d_ATR_VolatilityBreakout_MeanReversion_v1"
timeframe = "12h"
leverage = 1.0