#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d EMA50 trend filter and volume confirmation.
# Donchian channel breakouts capture breakout moves in the direction of the trend.
# EMA50 on daily ensures we trade with the higher timeframe trend.
# Volume confirmation ensures breakouts are supported by participation.
# Works in bull/bear markets: avoids false breakouts in ranging markets, captures true breakouts.
# Target: 20-50 trades/year per symbol.
name = "4h_Donchian20_EMA50_Volume_Breakout"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA50 on daily
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Donchian Channel (20) on 4h
    donch_period = 20
    upper_donch = pd.Series(high).rolling(window=donch_period, min_periods=donch_period).max().values
    lower_donch = pd.Series(low).rolling(window=donch_period, min_periods=donch_period).min().values
    
    # Align 1d EMA50 to 4h
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(donch_period, 50, 20)  # Ensure Donchian, EMA50, and volume MA are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_donch[i]) or np.isnan(lower_donch[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper = upper_donch[i]
        lower = lower_donch[i]
        ema_50_val = ema_50_aligned[i]
        vol_ma = vol_ma_20[i]
        vol = volume[i]
        
        # Volume confirmation threshold
        volume_confirmed = vol > 1.5 * vol_ma
        
        # Breakout conditions
        bullish_breakout = price > upper  # Price breaks above upper Donchian band
        bearish_breakout = price < lower  # Price breaks below lower Donchian band
        
        if position == 0:
            # Look for entry in direction of daily trend with volume confirmation
            if bullish_breakout and (price > ema_50_val) and volume_confirmed:
                signals[i] = 0.25
                position = 1
            elif bearish_breakout and (price < ema_50_val) and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long when price returns to the lower Donchian band (mean reversion)
            if price < lower_donch[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short when price returns to the upper Donchian band
            if price > upper_donch[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals