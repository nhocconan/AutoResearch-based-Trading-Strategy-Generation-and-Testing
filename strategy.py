# 4h_SR_Trend_Breakout_V1
# Strategy: 4-hour Support/Resistance Breakout with Trend Filter and Volume Confirmation
# Long: Price breaks above daily resistance (highest high of last 20 days) in uptrend with volume surge
# Short: Price breaks below daily support (lowest low of last 20 days) in downtrend with volume surge
# Exit: Trend reversal (EMA cross) or opposite breakout
# Designed for 4h timeframe: ~20-50 trades/year per symbol (80-200 total over 4 years)
# Works in bull/bear via trend filter and breakout logic with volume confirmation

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for support/resistance levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Daily support/resistance (20-day high/low)
    resistance = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    support = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Daily EMA25 and EMA50 for trend filter (shorter for faster adaptation)
    ema_25_1d = pd.Series(close_1d).ewm(span=25, adjust=False, min_periods=25).mean().values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Daily volume average (20-period)
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all daily data to 4h timeframe
    resistance_aligned = align_htf_to_ltf(prices, df_1d, resistance)
    support_aligned = align_htf_to_ltf(prices, df_1d, support)
    ema_25_aligned = align_htf_to_ltf(prices, df_1d, ema_25_1d)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # need enough for EMA50
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(resistance_aligned[i]) or np.isnan(support_aligned[i]) or 
            np.isnan(ema_25_aligned[i]) or np.isnan(ema_50_aligned[i]) or
            np.isnan(vol_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend conditions
        uptrend = ema_25_aligned[i] > ema_50_aligned[i]
        downtrend = ema_25_aligned[i] < ema_50_aligned[i]
        
        # Volume confirmation (at least 1.5x average)
        vol_confirm = volume[i] > 1.5 * vol_ma_aligned[i]
        
        # Breakout conditions
        breakout_long = close[i] > resistance_aligned[i]
        breakout_short = close[i] < support_aligned[i]
        
        if position == 0:
            # Long: uptrend + volume + breakout above resistance
            if uptrend and vol_confirm and breakout_long:
                signals[i] = 0.25
                position = 1
            # Short: downtrend + volume + breakout below support
            elif downtrend and vol_confirm and breakout_short:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: trend reversal or breakdown below support
            if not uptrend or breakout_short:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: trend reversal or breakout above resistance
            if not downtrend or breakout_long:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_SR_Trend_Breakout_V1"
timeframe = "4h"
leverage = 1.0