#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h ATR-based volatility breakout with 12h trend filter and volume confirmation.
# Uses ATR(14) to detect volatility expansion from consolidation.
# 12h EMA(50) defines trend direction to avoid counter-trend trades.
# Volume spike confirms breakout validity.
# Designed for 6h timeframe to capture medium-term volatility breakouts with low frequency.
# Entry: Long when price > upper band (MA + k*ATR) in uptrend with volume spike.
#        Short when price < lower band (MA - k*ATR) in downtrend with volume spike.
# Exit: Opposite band touch or trend reversal.
# Uses strict conditions to limit trades (~15-25/year) and avoid overtrading.
name = "6h_ATR_Volatility_Breakout_Trend"
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
    
    # 12h EMA(50) for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # ATR(14) calculation
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # 6-period moving average for volatility bands
    ma_6 = pd.Series(close).rolling(window=6, min_periods=6).mean().values
    
    # Volatility bands: upper = MA + k*ATR, lower = MA - k*ATR
    k = 2.0
    upper_band = ma_6 + k * atr
    lower_band = ma_6 - k * atr
    
    # Volume spike: volume > 2.0 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or 
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above upper band in uptrend with volume spike
            if (close[i] > upper_band[i] and 
                close[i] > ema_50_12h_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: break below lower band in downtrend with volume spike
            elif (close[i] < lower_band[i] and 
                  close[i] < ema_50_12h_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price touches lower band or trend turns down
            if (close[i] < lower_band[i]) or (close[i] < ema_50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price touches upper band or trend turns up
            if (close[i] > upper_band[i]) or (close[i] > ema_50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals