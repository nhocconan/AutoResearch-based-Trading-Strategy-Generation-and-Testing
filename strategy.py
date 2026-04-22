#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Hypothesis: Daily Bollinger Band squeeze breakout with weekly EMA34 trend and volume confirmation
    # Works in both bull and bear markets: breakouts from low volatility capture directional moves
    # Bollinger Band squeeze identifies compression before expansion
    # Volume surge confirms breakout strength, weekly EMA34 filters trend direction
    
    # Load daily data once
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Weekly EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_1w_34 = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1w_34_aligned = align_htf_to_ltf(prices, df_1w, ema_1w_34)
    
    # Daily Bollinger Bands (20, 2)
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate BB middle (SMA20), upper and lower bands
    sma20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_band = sma20 + 2 * std20
    lower_band = sma20 - 2 * std20
    
    # Bollinger Band width (normalized)
    bb_width = (upper_band - lower_band) / sma20
    
    # Bollinger Band squeeze: BB width below 20-period mean (low volatility)
    bb_width_ma20 = pd.Series(bb_width).rolling(window=20, min_periods=20).mean().values
    bb_squeeze = bb_width < bb_width_ma20
    
    # Volume filter (20-period MA surge)
    vol_ma20 = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_surge = prices['volume'].values > 1.5 * vol_ma20
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(200, n):
        # Skip if data not ready
        if (np.isnan(ema_1w_34_aligned[i]) or np.isnan(sma20[i]) or 
            np.isnan(std20[i]) or np.isnan(vol_ma20[i]) or np.isnan(bb_width_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: BB squeeze breakout above upper band with volume surge AND weekly EMA34 uptrend
            if close[i] > upper_band[i] and bb_squeeze[i] and vol_surge[i] and close[i] > ema_1w_34_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: BB squeeze breakout below lower band with volume surge AND weekly EMA34 downtrend
            elif close[i] < lower_band[i] and bb_squeeze[i] and vol_surge[i] and close[i] < ema_1w_34_aligned[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price returns to Bollinger middle (SMA20) or opposite band touch
            if position == 1:
                if close[i] < sma20[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > sma20[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "1d_Bollinger_Squeeze_Breakout_1wEMA34_Trend_VolumeSurge_v1"
timeframe = "1d"
leverage = 1.0