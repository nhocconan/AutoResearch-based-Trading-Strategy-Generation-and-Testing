#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R4/S4 breakout with 1w EMA50 trend filter and volume confirmation
# Long when price breaks above R4 AND close > 1w EMA50 (uptrend) AND volume > 1.8 * 20-bar avg volume
# Short when price breaks below S4 AND close < 1w EMA50 (downtrend) AND volume > 1.8 * 20-bar avg volume
# Exit when price retouches the Camarilla pivot level (mean reversion to equilibrium)
# Uses discrete sizing 0.28 to balance return and fee drag
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe
# Camarilla R4/S4 levels provide stronger breakout signals than R3/S3, reducing false breakouts
# 1w EMA50 ensures alignment with weekly trend for better survivability in bear markets
# Volume spike filters out low-participation moves; pivot retouch exit works in ranging markets
# Designed for BTC/ETH primary focus with SOL as secondary validation

name = "4h_Camarilla_R4S4_1wEMA50_Volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla levels for 4h timeframe (based on previous bar)
    ph = np.concatenate([[high[0]], high[:-1]])  # previous high
    pl = np.concatenate([[low[0]], low[:-1]])    # previous low
    pc = np.concatenate([[close[0]], close[:-1]]) # previous close
    
    pivot = (ph + pl + pc) / 3.0
    range_ = ph - pl
    
    # Camarilla levels (R4/S4 = stronger breakout thresholds)
    r4 = pivot + (range_ * 1.1 / 2.0)
    s4 = pivot - (range_ * 1.1 / 2.0)
    
    # Get 1w data ONCE before loop for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA50
    close_1w_series = pd.Series(close_1w)
    ema50_1w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF indicators to 4h timeframe (wait for completed HTF bar)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate volume confirmation: volume > 1.8 * 20-bar average volume
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(r4[i]) or np.isnan(s4[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Camarilla R4/S4 breakout signals with trend and volume filters
            # Long: Break above R4 AND uptrend AND volume spike
            if close[i] > r4[i] and close[i] > ema50_1w_aligned[i] and volume_spike[i]:
                signals[i] = 0.28
                position = 1
            # Short: Break below S4 AND downtrend AND volume spike
            elif close[i] < s4[i] and close[i] < ema50_1w_aligned[i] and volume_spike[i]:
                signals[i] = -0.28
                position = -1
        elif position == 1:
            # Exit long: Price retouches pivot level (mean reversion)
            if close[i] <= pivot[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.28
        elif position == -1:
            # Exit short: Price retouches pivot level (mean reversion)
            if close[i] >= pivot[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.28
    
    return signals