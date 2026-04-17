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
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA34 for trend filter
    ema34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    # Get 1d data for daily volatility filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily ATR for volatility regime filter
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = np.nan
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1d_ma10 = pd.Series(atr_1d).rolling(window=10, min_periods=10).mean().values
    
    # Align daily ATR and its MA to 12h timeframe
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    atr_1d_ma10_aligned = align_htf_to_ltf(prices, df_1d, atr_1d_ma10)
    
    # Calculate 12h ATR for stoploss and position sizing
    tr1_12h = high - low
    tr2_12h = np.abs(high - np.roll(close, 1))
    tr3_12h = np.abs(low - np.roll(close, 1))
    tr1_12h[0] = np.nan
    tr2_12h[0] = np.nan
    tr3_12h[0] = np.nan
    tr_12h = np.maximum(tr1_12h, np.maximum(tr2_12h, tr3_12h))
    atr_12h = pd.Series(tr_12h).rolling(window=14, min_periods=14).mean().values
    
    # 4-period volume average for volume confirmation (12h * 4 = 2 days)
    volume_ma4 = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 34  # Need EMA34 and ATR data
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema34_12h_aligned[i]) or 
            np.isnan(atr_1d_aligned[i]) or 
            np.isnan(atr_1d_ma10_aligned[i]) or
            np.isnan(volume_ma4[i]) or
            np.isnan(atr_12h[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 12h EMA34
        uptrend = close[i] > ema34_12h_aligned[i]
        downtrend = close[i] < ema34_12h_aligned[i]
        
        # Volatility filter: daily ATR > its 10-period mean (avoid low volatility regimes)
        volatility_filter = atr_1d_aligned[i] > atr_1d_ma10_aligned[i]
        
        # Volume filter: current volume > 1.3x 4-period average
        volume_filter = volume[i] > (1.3 * volume_ma4[i])
        
        if position == 0:
            # Long: uptrend + volatility + volume + price above EMA34
            if uptrend and volatility_filter and volume_filter:
                signals[i] = 0.25
                position = 1
            # Short: downtrend + volatility + volume + price below EMA34
            elif downtrend and volatility_filter and volume_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: trend reversal or volatility drop
            if not uptrend or not volatility_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: trend reversal or volatility drop
            if not downtrend or not volatility_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_EMA34_Trend_Volume_Volatility_Filter"
timeframe = "12h"
leverage = 1.0