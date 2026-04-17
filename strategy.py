#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for higher timeframe trend
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # 12h EMA(34) for trend filter
    ema34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    # Calculate 12h ATR for volatility regime
    tr1_12h = high_12h - low_12h
    tr2_12h = np.abs(high_12h - np.roll(close_12h, 1))
    tr3_12h = np.abs(low_12h - np.roll(close_12h, 1))
    tr1_12h[0] = np.nan
    tr2_12h[0] = np.nan
    tr3_12h[0] = np.nan
    tr_12h = np.maximum(tr1_12h, np.maximum(tr2_12h, tr3_12h))
    atr_12h = pd.Series(tr_12h).rolling(window=14, min_periods=14).mean().values
    atr_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_12h)
    
    # 6h ATR for position sizing and stops
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = np.nan
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 6h Bollinger Bands (20, 2)
    sma20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_band = sma20 + (2 * std20)
    lower_band = sma20 - (2 * std20)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 40  # Need EMA34, ATR, Bollinger Bands
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema34_12h_aligned[i]) or 
            np.isnan(atr_12h_aligned[i]) or 
            np.isnan(atr[i]) or
            np.isnan(sma20[i]) or 
            np.isnan(std20[i]) or
            np.isnan(upper_band[i]) or 
            np.isnan(lower_band[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: 12h ATR > 12h ATR MA50 (high volatility regime)
        atr_ma50_12h = pd.Series(atr_12h_aligned).rolling(window=50, min_periods=50).mean().values
        if np.isnan(atr_ma50_12h[i]):
            signals[i] = 0.0
            continue
        high_vol_regime = atr_12h_aligned[i] > atr_ma50_12h[i]
        
        # Bollinger Band position
        bb_position = (close[i] - lower_band[i]) / (upper_band[i] - lower_band[i])
        
        if position == 0:
            # Long: 12h uptrend + high volatility + price near lower BB (mean reversion in trend)
            if (ema34_12h_aligned[i] > ema34_12h_aligned[i-1] and  # 12h EMA rising
                high_vol_regime and 
                bb_position < 0.2):  # Near lower band
                signals[i] = 0.25
                position = 1
            # Short: 12h downtrend + high volatility + price near upper BB
            elif (ema34_12h_aligned[i] < ema34_12h_aligned[i-1] and  # 12h EMA falling
                  high_vol_regime and 
                  bb_position > 0.8):  # Near upper band
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: 12h trend reversal or price reaches middle/lower BB
            if (ema34_12h_aligned[i] < ema34_12h_aligned[i-1] or  # 12h EMA turned down
                bb_position > 0.5):  # Back to middle or upper band
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: 12h trend reversal or price reaches middle/upper BB
            if (ema34_12h_aligned[i] > ema34_12h_aligned[i-1] or  # 12h EMA turned up
                bb_position < 0.5):  # Back to middle or lower band
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_12hEMA34_BB_MeanReversion_Trend"
timeframe = "6h"
leverage = 1.0