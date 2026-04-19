#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h trading based on 1-day Camarilla pivot points (H3/L3) with volume confirmation and RSI momentum filter.
# Camarilla H3/L3 act as key support/resistance levels derived from previous day's range.
# Breakouts above H3 with volume and bullish RSI indicate strong momentum; breakdowns below L3 with volume and bearish RSI indicate strong downside.
# RSI filter avoids entries in choppy conditions. Designed for low-frequency, high-conviction trades (target: 15-25/year).
# Works in both bull and bear markets by capturing genuine breakouts with volume confirmation.
name = "12h_Camarilla_H3L3_Volume_RSI"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1-day data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla levels
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Calculate Camarilla H3 and L3 levels
    # H3 = close + (high - low) * 1.1 / 2
    # L3 = close - (high - low) * 1.1 / 2
    camarilla_h3 = prev_close + (prev_high - prev_low) * 1.1 / 2
    camarilla_l3 = prev_close - (prev_high - prev_low) * 1.1 / 2
    
    # Align Camarilla levels to 12h timeframe (ensures use of prior day's close)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # RSI(14) for momentum filter
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # Volume spike: volume > 2.0 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure sufficient data for indicators
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(rsi[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: price breaks above H3 with bullish RSI and volume spike
            if (close[i] > camarilla_h3_aligned[i] and 
                rsi[i] > 50 and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below L3 with bearish RSI and volume spike
            elif (close[i] < camarilla_l3_aligned[i] and 
                  rsi[i] < 50 and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long exit: price touches L3 or RSI turns bearish
            if (close[i] < camarilla_l3_aligned[i]) or (rsi[i] < 40):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short exit: price touches H3 or RSI turns bullish
            if (close[i] > camarilla_h3_aligned[i]) or (rsi[i] > 60):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals