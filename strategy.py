#!/usr/bin/env python3
"""
12h_3ATR_Breakout_Trend_Volume
12h strategy using 3x ATR breakout from ATR-based channels with trend and volume filters.
Long: Close breaks above upper band (mean + 3*ATR) + trend filter + volume confirmation
Short: Close breaks below lower band (mean - 3*ATR) + trend filter + volume confirmation
Exit: Opposite breakout or trend reversal
Designed for ~20-30 trades/year per symbol (80-120 total over 4 years)
ATR channels provide volatility-adaptive breakout levels that work in both bull and bear markets
"""

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
    
    # Calculate ATR-based channels on primary timeframe
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    atr = np.zeros_like(tr)
    atr[0] = tr[0]
    for i in range(1, len(tr)):
        atr[i] = 0.9 * atr[i-1] + 0.1 * tr[i]  # Wilder's smoothing
    
    # Calculate mean price (close) for channel center
    mean_price = close
    
    # Upper and lower bands (mean ± 3*ATR)
    upper_band = mean_price + 3.0 * atr
    lower_band = mean_price - 3.0 * atr
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Daily EMA50 and EMA200 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Daily volume average (20-period)
    volume_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all daily data to 12h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # need enough for EMA200
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_50_aligned[i]) or np.isnan(ema_200_aligned[i]) or
            np.isnan(vol_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend conditions
        uptrend = ema_50_aligned[i] > ema_200_aligned[i]
        downtrend = ema_50_aligned[i] < ema_200_aligned[i]
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma_aligned[i]
        
        # Breakout conditions
        breakout_up = close[i] > upper_band[i]
        breakdown_down = close[i] < lower_band[i]
        
        if position == 0:
            # Long: uptrend + volume + breakout above upper band
            if uptrend and vol_confirm and breakout_up:
                signals[i] = 0.25
                position = 1
            # Short: downtrend + volume + breakdown below lower band
            elif downtrend and vol_confirm and breakdown_down:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: trend change, volume confirmation, or breakdown below lower band
            if not uptrend or (vol_confirm and breakdown_down):
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: trend change, volume confirmation, or breakout above upper band
            if not downtrend or (vol_confirm and breakout_up):
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_3ATR_Breakout_Trend_Volume"
timeframe = "12h"
leverage = 1.0