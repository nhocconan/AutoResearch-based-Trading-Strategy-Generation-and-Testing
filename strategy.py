#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1-day Parabolic SAR to capture trend direction,
# confirmed by 6-hour Bollinger Band breakout with volume spike.
# Parabolic SAR provides clear trend signals with built-in acceleration.
# Bollinger Band breakouts with volume capture momentum bursts.
# Works in both bull/bear markets by requiring alignment between daily trend
# and 6h momentum breakout direction.
# Target: 20-30 trades/year per symbol.
name = "6h_PSAR_1d_BB20_Breakout_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily Parabolic SAR for trend direction
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Parabolic SAR calculation
    psar = np.zeros(len(high_1d))
    psar[0] = low_1d[0]
    bull = True  # Start with bullish assumption
    af = 0.02    # Acceleration factor
    max_af = 0.2
    ep = high_1d[0] if bull else low_1d[0]  # Extreme point
    
    for i in range(1, len(high_1d)):
        if bull:
            psar[i] = psar[i-1] + af * (ep - psar[i-1])
            if low_1d[i] < psar[i]:
                bull = False
                psar[i] = ep
                af = 0.02
                ep = low_1d[i]
            else:
                if high_1d[i] > ep:
                    ep = high_1d[i]
                    af = min(af + 0.02, max_af)
        else:
            psar[i] = psar[i-1] + af * (ep - psar[i-1])
            if high_1d[i] > psar[i]:
                bull = True
                psar[i] = ep
                af = 0.02
                ep = high_1d[i]
            else:
                if low_1d[i] < ep:
                    ep = low_1d[i]
                    af = min(af + 0.02, max_af)
    
    psar_aligned = align_htf_to_ltf(prices, df_1d, psar)
    
    # 6-hour Bollinger Bands (20, 2)
    bb_period = 20
    bb_std = 2
    sma = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    std_dev = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper_band = sma + (std_dev * bb_std)
    lower_band = sma - (std_dev * bb_std)
    
    # Volume spike: volume > 2.0 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(bb_period, 20)  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(psar_aligned[i]) or np.isnan(sma[i]) or 
            np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or 
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above upper BB, above daily PSAR (bullish trend), with volume spike
            if (close[i] > upper_band[i] and 
                close[i] > psar_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower BB, below daily PSAR (bearish trend), with volume spike
            elif (close[i] < lower_band[i] and 
                  close[i] < psar_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks below lower BB or below daily PSAR
            if (close[i] < lower_band[i]) or (close[i] < psar_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price breaks above upper BB or above daily PSAR
            if (close[i] > upper_band[i]) or (close[i] > psar_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals