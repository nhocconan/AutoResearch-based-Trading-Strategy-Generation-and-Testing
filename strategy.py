#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot (from 1d) with volume confirmation and momentum filter
# Long when price closes above R3 with volume > 1.5x avg and RSI(14) > 50
# Short when price closes below S3 with volume > 1.5x avg and RSI(14) < 50
# Exit when price crosses H4/L4 levels or RSI crosses 50 in opposite direction
# Uses 6h timeframe targeting 50-150 total trades over 4 years (12-37/year)
# Camarilla levels act as dynamic support/resistance; volume confirms institutional interest
# RSI filter ensures momentum alignment, reducing false breakouts in chop

name = "6h_camarilla_rsi_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla calculation (call once before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels from previous 1d bar
    # H4 = close + 1.1*(high-low)*1.1/2, L4 = close - 1.1*(high-low)*1.1/2
    # H3 = close + 1.1*(high-low)*1.1/4, L3 = close - 1.1*(high-low)*1.1/4
    # H2 = close + 1.1*(high-low)*1.1/6, L2 = close - 1.1*(high-low)*1.1/6
    # H1 = close + 1.1*(high-low)*1.1/12, L1 = close - 1.1*(high-low)*1.1/12
    camarilla_h4 = np.zeros_like(close_1d)
    camarilla_l4 = np.zeros_like(close_1d)
    camarilla_h3 = np.zeros_like(close_1d)
    camarilla_l3 = np.zeros_like(close_1d)
    
    for i in range(1, len(close_1d)):
        rang = high_1d[i-1] - low_1d[i-1]
        camarilla_h4[i] = close_1d[i-1] + 1.1 * rang * 1.1 / 2
        camarilla_l4[i] = close_1d[i-1] - 1.1 * rang * 1.1 / 2
        camarilla_h3[i] = close_1d[i-1] + 1.1 * rang * 1.1 / 4
        camarilla_l3[i] = close_1d[i-1] - 1.1 * rang * 1.1 / 4
    
    # Align Camarilla levels to 6h timeframe
    h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # RSI(14) for momentum filter
    def rsi(close_prices, period=14):
        delta = np.diff(close_prices, prepend=close_prices[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.zeros_like(close_prices)
        avg_loss = np.zeros_like(close_prices)
        
        for i in range(1, len(close_prices)):
            if i < period:
                avg_gain[i] = np.mean(gain[1:i+1]) if i > 0 else 0
                avg_loss[i] = np.mean(loss[1:i+1]) if i > 0 else 0
            else:
                avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
                avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
        
        rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
        rsi_vals = 100 - (100 / (1 + rs))
        return rsi_vals
    
    rsi_vals = rsi(close, 14)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or 
            np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or
            np.isnan(rsi_vals[i]) or np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Exit conditions
        if position == 1:  # long position
            if close[i] <= l4_aligned[i] or rsi_vals[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if close[i] >= h4_aligned[i] or rsi_vals[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume and momentum confirmation
            # Bullish: close above H3 with volume and RSI > 50
            if (close[i] > h3_aligned[i] and 
                volume[i] > volume_threshold[i] and 
                rsi_vals[i] > 50):
                signals[i] = 0.25
                position = 1
            # Bearish: close below L3 with volume and RSI < 50
            elif (close[i] < l3_aligned[i] and 
                  volume[i] > volume_threshold[i] and 
                  rsi_vals[i] < 50):
                signals[i] = -0.25
                position = -1
    
    return signals