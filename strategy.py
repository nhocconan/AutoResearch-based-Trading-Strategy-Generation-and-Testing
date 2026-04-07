#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 4h Donchian(20) breakout with weekly volume confirmation and momentum filter
# Hypothesis: Breakouts with higher timeframe volume confirmation capture strong trends, while momentum filter avoids false breakouts in chop.
# Works in bull via breakouts, in bear via momentum-based mean reversion when price reverts to mean after overextension.
# Target: 20-50 trades/year to minimize fee drag.
name = "4h_donchian20_1w_volume_mom_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for volume confirmation and momentum
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate weekly 20-period volume moving average
    vol_1w = df_1w['volume'].values
    vol_ma_1w = pd.Series(vol_1w).rolling(window=20, min_periods=20).mean().values
    vol_ma_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_1w)
    
    # Calculate weekly RSI(14) for momentum filter
    delta = pd.Series(df_1w['close']).diff().values
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi_1w = 100 - (100 / (1 + rs))
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    
    # Calculate ATR(14) for volatility filter and position sizing
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Donchian channels (20-period high/low)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(vol_ma_1w_aligned[i]) or np.isnan(rsi_1w_aligned[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4h volume > weekly average volume
        vol_confirm = volume[i] > vol_ma_1w_aligned[i]
        
        # Momentum filter: only trade when weekly RSI is not extreme (avoid overextended moves)
        mom_filter = (rsi_1w_aligned[i] > 30) and (rsi_1w_aligned[i] < 70)
        
        if position == 1:  # Long position
            # Exit: price touches opposite band OR momentum turns bearish
            if close[i] <= lowest_low[i] or rsi_1w_aligned[i] < 30:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: price touches opposite band OR momentum turns bullish
            if close[i] >= highest_high[i] or rsi_1w_aligned[i] > 70:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Enter long: price breaks above upper band + volume confirmation + momentum filter
            if close[i] > highest_high[i] and vol_confirm and mom_filter:
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below lower band + volume confirmation + momentum filter
            elif close[i] < lowest_low[i] and vol_confirm and mom_filter:
                position = -1
                signals[i] = -0.25
    
    return signals