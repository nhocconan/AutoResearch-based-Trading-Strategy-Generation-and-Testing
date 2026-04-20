#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Choppiness Index + 1w RSI Mean Reversion
# - Uses Choppiness Index (14) on 12h to detect ranging markets (CHOP > 61.8)
# - In ranging markets, uses 1-week RSI(14) for mean reversion: long when RSI < 30, short when RSI > 70
# - Filters out trending markets (CHOP < 38.2) to avoid false signals
# - Volume confirmation: current volume > 1.2x 20-period average
# - Designed for 12h timeframe with low frequency to avoid overtrading
# - Target: 12-37 trades per year per symbol (50-150 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1w data for RSI calculation
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate RSI(14) on 1w timeframe
    delta = pd.Series(close_1w).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_1w = (100 - (100 / (1 + rs))).values
    
    # Align 1w RSI to 12h timeframe
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    
    # Calculate Choppiness Index on 12h timeframe
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # ATR(14)
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Sum of ATR over 14 periods
    sum_atr = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index
    chop = 100 * np.log10(sum_atr / (highest_high - lowest_low)) / np.log10(14)
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(14, n):  # Start after Chop/RSI warmup
        # Skip if NaN in indicators
        if np.isnan(chop[i]) or np.isnan(rsi_1w_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        
        if position == 0:
            # Only trade in ranging markets (CHOP > 61.8)
            if chop[i] > 61.8:
                # Long entry: RSI < 30 (oversold) + volume confirmation
                if rsi_1w_aligned[i] < 30 and vol > 1.2 * vol_ma[i]:
                    signals[i] = 0.25
                    position = 1
                # Short entry: RSI > 70 (overbought) + volume confirmation
                elif rsi_1w_aligned[i] > 70 and vol > 1.2 * vol_ma[i]:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Long exit: RSI > 50 (mean reversion complete) or market becomes trending
            if rsi_1w_aligned[i] > 50 or chop[i] < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI < 50 (mean reversion complete) or market becomes trending
            if rsi_1w_aligned[i] < 50 or chop[i] < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Chop_1wRSI_MeanReversion"
timeframe = "12h"
leverage = 1.0