#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R mean reversion with 1d trend filter and volume confirmation.
# Williams %R identifies overbought/oversold conditions. We trade reversals from extremes
# in the direction of the daily trend (EMA50) with volume confirmation.
# Works in bull/bear markets: avoids counter-trend trades, captures mean reversion in trends.
# Target: 20-40 trades/year per symbol.
name = "4h_WilliamsR_EMA50_Volume_MeanReversion"
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
    
    # Williams %R (14) on 4h
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # Align 1d EMA50 to 4h
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(14, 50, 20)  # Ensure Williams %R, EMA50, and volume MA are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(williams_r[i]) or
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        wr = williams_r[i]
        ema_50_val = ema_50_aligned[i]
        vol_ma = vol_ma_20[i]
        vol = volume[i]
        
        # Volume confirmation threshold
        volume_confirmed = vol > 1.5 * vol_ma
        
        # Williams %R levels
        oversold = wr < -80  # Oversold condition
        overbought = wr > -20  # Overbought condition
        
        if position == 0:
            # Look for reversal from extremes in direction of daily trend
            if oversold and (close[i] > ema_50_val) and volume_confirmed:
                signals[i] = 0.25
                position = 1
            elif overbought and (close[i] < ema_50_val) and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long when Williams %R returns to neutral territory or trend weakens
            if wr > -50:  # Return to neutral or overbought
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short when Williams %R returns to neutral territory
            if wr < -50:  # Return to neutral or oversold
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals