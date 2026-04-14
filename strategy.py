#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d RSI(14) for momentum
    close_series = pd.Series(df_1d['close'])
    delta = close_series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_14_1d = (100 - (100 / (1 + rs))).values
    rsi_14_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_14_1d)
    
    # Calculate 1d ATR(14) for volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate 1d Bollinger Bands (20, 2)
    sma_20_1d = close_series.rolling(window=20, min_periods=20).mean().values
    std_20_1d = close_series.rolling(window=20, min_periods=20).std().values
    upper_bb_1d = sma_20_1d + (2 * std_20_1d)
    lower_bb_1d = sma_20_1d - (2 * std_20_1d)
    upper_bb_aligned = align_htf_to_ltf(prices, df_1d, upper_bb_1d)
    lower_bb_aligned = align_htf_to_ltf(prices, df_1d, lower_bb_1d)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 100
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(rsi_14_1d_aligned[i]) or 
            np.isnan(atr_14_1d_aligned[i]) or 
            np.isnan(upper_bb_aligned[i]) or 
            np.isnan(lower_bb_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        # Volatility filter: ATR > 0.5% of price
        atr_ratio = atr_14_1d_aligned[i] / price if price > 0 else 0
        vol_filter = atr_ratio > 0.005
        
        # Bollinger Band squeeze detection: bandwidth < 4% of price
        bb_width = (upper_bb_aligned[i] - lower_bb_aligned[i]) / price if price > 0 else 0
        bb_squeeze = bb_width < 0.04
        
        # RSI momentum filter: RSI > 55 for long, RSI < 45 for short
        rsi_long_filter = rsi_14_1d_aligned[i] > 55
        rsi_short_filter = rsi_14_1d_aligned[i] < 45
        
        if position == 0:
            # Long setup: RSI > 55 + volatility filter + not in squeeze
            if (rsi_long_filter and vol_filter and not bb_squeeze):
                position = 1
                signals[i] = position_size
            # Short setup: RSI < 45 + volatility filter + not in squeeze
            elif (rsi_short_filter and vol_filter and not bb_squeeze):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: RSI drops below 50
            if rsi_14_1d_aligned[i] < 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: RSI rises above 50
            if rsi_14_1d_aligned[i] > 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1dRSI14_BBWidth_Filter"
timeframe = "12h"
leverage = 1.0