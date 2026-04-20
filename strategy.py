#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Choppiness Index regime filter with 12h Williams Alligator trend
# Choppiness Index > 61.8 = ranging market (mean reversion), < 38.2 = trending (trend follow)
# Williams Alligator (Jaw/Teeth/Lips) provides trend direction in trending regimes
# In ranging markets: fade extreme RSI moves (overbought/oversold)
# In trending markets: trade Alligator crossovers with trend filter
# Designed to reduce whipsaw in ranging markets and capture trends when present

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Williams Alligator on 12h: Jaw (13), Teeth (8), Lips (5) SMAs
    jaw_12h = pd.Series(close_12h).rolling(window=13, min_periods=13).mean().values
    teeth_12h = pd.Series(close_12h).rolling(window=8, min_periods=8).mean().values
    lips_12h = pd.Series(close_12h).rolling(window=5, min_periods=5).mean().values
    
    jaw_12h_aligned = align_htf_to_ltf(prices, df_12h, jaw_12h)
    teeth_12h_aligned = align_htf_to_ltf(prices, df_12h, teeth_12h)
    lips_12h_aligned = align_htf_to_ltf(prices, df_12h, lips_12h)
    
    # Calculate 14-period Choppiness Index
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Sum of True Range over 14 periods
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index
    chi = 100 * np.log10(tr_sum / (hh - ll)) / np.log10(14)
    
    # RSI for mean reversion signals in ranging markets
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    # Prepend NaN for first element
    rsi = np.concatenate([[np.nan], rsi])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if NaN in indicators
        if np.isnan(chi[i]) or np.isnan(rsi[i]) or \
           np.isnan(jaw_12h_aligned[i]) or np.isnan(teeth_12h_aligned[i]) or np.isnan(lips_12h_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Regime detection
        is_ranging = chi[i] > 61.8
        is_trending = chi[i] < 38.2
        
        # Williams Alligator signals
        # Bullish: Lips > Teeth > Jaw (all aligned up)
        bullish_aligned = lips_12h_aligned[i] > teeth_12h_aligned[i] > jaw_12h_aligned[i]
        # Bearish: Lips < Teeth < Jaw (all aligned down)
        bearish_aligned = lips_12h_aligned[i] < teeth_12h_aligned[i] < jaw_12h_aligned[i]
        
        price = close[i]
        
        if position == 0:
            if is_ranging:
                # In ranging markets: mean reversion at RSI extremes
                long_signal = rsi[i] < 30  # Oversold
                short_signal = rsi[i] > 70  # Overbought
            elif is_trending:
                # In trending markets: follow Alligator alignment
                long_signal = bullish_aligned
                short_signal = bearish_aligned
            else:
                # Transition zone: no clear signal
                long_signal = False
                short_signal = False
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = price
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: opposite Alligator signal or RSI overbought in ranging
            if is_ranging and rsi[i] > 70:
                signals[i] = 0.0
                position = 0
            elif not is_ranging and not bullish_aligned:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: opposite Alligator signal or RSI oversold in ranging
            if is_ranging and rsi[i] < 30:
                signals[i] = 0.0
                position = 0
            elif not is_ranging and not bearish_aligned:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_ChopRegime_Alligator_RSI"
timeframe = "4h"
leverage = 1.0