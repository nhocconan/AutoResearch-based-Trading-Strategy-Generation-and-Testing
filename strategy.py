#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6s strategy using 1-day True Range bands (ATR-based) to identify volatility expansion,
# combined with 12-hour RSI for momentum and volume confirmation. In both bull and bear markets,
# volatility expansion often precedes directional moves. We enter long when price breaks above
# upper TR band with bullish RSI and volume spike, short when breaks below lower TR band with
# bearish RSI and volume spike. Uses ATR-based dynamic bands to adapt to changing volatility.
# Target: 20-40 trades/year per symbol with disciplined entries.
name = "6h_TRBands_RSI12h_Volume"
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
    
    # Daily True Range and ATR for volatility bands
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate True Range for daily data
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = np.abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate upper and lower bands (midpoint = daily close)
    daily_close = df_1d['close'].values
    upper_band = daily_close + (atr_14_1d * 1.5)
    lower_band = daily_close - (atr_14_1d * 1.5)
    
    # Align bands to 6t
    upper_band_aligned = align_htf_to_ltf(prices, df_1d, upper_band)
    lower_band_aligned = align_htf_to_ltf(prices, df_1d, lower_band)
    
    # 12-hour RSI for momentum
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 14:
        return np.zeros(n)
    
    delta = pd.Series(df_12h['close']).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_14_12h = (100 - (100 / (1 + rs))).values
    rsi_14_12h_aligned = align_htf_to_ltf(prices, df_12h, rsi_14_12h)
    
    # Volume spike: volume > 2.0 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_band_aligned[i]) or np.isnan(lower_band_aligned[i]) or 
            np.isnan(rsi_14_12h_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above upper TR band, RSI > 50 (bullish), volume spike
            if (close[i] > upper_band_aligned[i] and 
                rsi_14_12h_aligned[i] > 50 and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price below lower TR band, RSI < 50 (bearish), volume spike
            elif (close[i] < lower_band_aligned[i] and 
                  rsi_14_12h_aligned[i] < 50 and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks below lower TR band or RSI turns bearish
            if (close[i] < lower_band_aligned[i]) or (rsi_14_12h_aligned[i] < 50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price breaks above upper TR band or RSI turns bullish
            if (close[i] > upper_band_aligned[i]) or (rsi_14_12h_aligned[i] > 50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals