#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 1d Bollinger Band squeeze breakout with weekly EMA50 trend filter and volume confirmation
    # Bollinger Band squeeze (low volatility) precedes breakout moves
    # Direction determined by weekly EMA50 trend
    # Volume surge confirms breakout validity
    # Works in both bull and bear markets by following the higher timeframe trend
    
    # Price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    ema50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Bollinger Bands (20, 2) on daily
    sma20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_band = sma20 + 2 * std20
    lower_band = sma20 - 2 * std20
    bb_width = (upper_band - lower_band) / sma20
    
    # Bollinger Band squeeze: width below 20-period mean
    bb_width_ma20 = pd.Series(bb_width).rolling(window=20, min_periods=20).mean().values
    bb_squeeze = bb_width < bb_width_ma20
    
    # Volume confirmation: 20-period volume average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_surge = volume > 2.0 * vol_ma20  # Require 2x volume for confirmation
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(50, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(sma20[i]) or np.isnan(std20[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bollinger squeeze breakout + weekly uptrend + volume surge
            if bb_squeeze[i] and close[i] > upper_band[i] and close[i] > ema50_1w_aligned[i] and vol_surge[i]:
                signals[i] = 0.25
                position = 1
            # Short: Bollinger squeeze breakdown + weekly downtrend + volume surge
            elif bb_squeeze[i] and close[i] < lower_band[i] and close[i] < ema50_1w_aligned[i] and vol_surge[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price returns to middle Bollinger Band or trend reversal vs weekly EMA50
            if position == 1:
                if close[i] < sma20[i] or close[i] < ema50_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > sma20[i] or close[i] > ema50_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "1d_Bollinger_Squeeze_Breakout_1wEMA50_Trend_VolumeSurge_v1"
timeframe = "1d"
leverage = 1.0