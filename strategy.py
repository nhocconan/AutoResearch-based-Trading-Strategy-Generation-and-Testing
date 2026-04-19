#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with weekly volatility filter and volume confirmation.
# Uses weekly ATR to filter breakouts during low volatility periods, avoiding false breakouts in ranging markets.
# Trades breakouts in the direction of the breakout (not trend-following) with volume confirmation.
# Weekly volatility filter ensures we only trade when volatility is expanding, which works in both bull and bear markets.
# Target: 12-30 trades/year per symbol.
name = "12h_Donchian20_WeeklyVol_Filter_Volume"
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
    
    # Get weekly data for ATR filter
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly ATR(14)
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First TR is just high-low
    atr_14_1w = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Donchian channels (20-period) on 12h
    donch_period = 20
    upper_donch = pd.Series(high).rolling(window=donch_period, min_periods=donch_period).max().values
    lower_donch = pd.Series(low).rolling(window=donch_period, min_periods=donch_period).min().values
    
    # Align weekly ATR to 12h
    atr_14_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_14_1w)
    
    # Volatility filter: weekly ATR > 50-period average of weekly ATR (expanding volatility)
    atr_ma_50 = pd.Series(atr_14_1w_aligned).rolling(window=50, min_periods=50).mean().values
    vol_expanding = atr_14_1w_aligned > atr_ma_50
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(donch_period, 50, 20)  # Ensure Donchian, ATR MA, and volume MA are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_donch[i]) or np.isnan(lower_donch[i]) or 
            np.isnan(atr_14_1w_aligned[i]) or np.isnan(atr_ma_50[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper = upper_donch[i]
        lower = lower_donch[i]
        vol_exp = vol_expanding[i]
        vol_ma = vol_ma_20[i]
        vol = volume[i]
        
        # Volume confirmation threshold
        volume_confirmed = vol > 1.5 * vol_ma
        
        # Breakout conditions
        bullish_breakout = price > upper  # Price breaks above upper Donchian
        bearish_breakout = price < lower  # Price breaks below lower Donchian
        
        if position == 0:
            # Look for breakout with expanding volatility and volume confirmation
            if bullish_breakout and vol_exp and volume_confirmed:
                signals[i] = 0.25
                position = 1
            elif bearish_breakout and vol_exp and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long when price returns to the lower Donchian channel (mean reversion)
            if price < lower_donch[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short when price returns to the upper Donchian channel
            if price > upper_donch[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals