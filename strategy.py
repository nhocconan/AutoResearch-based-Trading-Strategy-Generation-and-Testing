#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h 1-week RSI divergence with volume confirmation and weekly trend filter.
# Uses weekly RSI extremes (>80 or <20) for mean reversion entries, confirmed by volume spike.
# Weekly EMA20 trend filter ensures trades align with higher timeframe momentum.
# Designed for low frequency (~15-25 trades/year) to minimize fee drag while capturing
# mean reversion extremes in both bull and bear markets.
name = "12h_WeeklyRSI_Divergence_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Weekly RSI(14)
    delta = pd.Series(df_1w['close']).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_1w = (100 - (100 / (1 + rs))).values
    
    # Weekly EMA20 trend filter
    ema20_1w = pd.Series(df_1w['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align weekly indicators to 12h timeframe
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Volume confirmation: 2x 20-period average on 12h chart
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(rsi_1w_aligned[i]) or np.isnan(ema20_1w_aligned[i]) or
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Weekly RSI oversold (<20) + price above weekly EMA20 + volume spike
            if (rsi_1w_aligned[i] < 20 and 
                close[i] > ema20_1w_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Weekly RSI overbought (>80) + price below weekly EMA20 + volume spike
            elif (rsi_1w_aligned[i] > 80 and 
                  close[i] < ema20_1w_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if RSI returns to neutral (>40) or weekly trend turns bearish
            if (rsi_1w_aligned[i] > 40) or (close[i] < ema20_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if RSI returns to neutral (<60) or weekly trend turns bullish
            if (rsi_1w_aligned[i] < 60) or (close[i] > ema20_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals