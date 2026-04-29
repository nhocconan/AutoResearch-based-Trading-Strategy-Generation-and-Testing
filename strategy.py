#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Bollinger Band breakout with 1d ATR volatility filter and volume confirmation
# Long when price breaks above upper BB(20,2) AND ATR(14) > 1.2*ATR(50) AND volume > 1.5x 20-bar avg
# Short when price breaks below lower BB(20,2) AND ATR(14) > 1.2*ATR(50) AND volume > 1.5x 20-bar avg
# Exit when price crosses middle BB(20) line (mean reversion)
# Uses discrete position sizing (0.25) to balance capture and risk.
# Bollinger Bands provide dynamic support/resistance that adapts to volatility.
# ATR filter ensures we only trade during sufficient volatility periods.
# Volume confirmation ensures participation.
# Target: 20-50 trades/year on 4h timeframe (80-200 total over 4 years) to avoid overtrading.

name = "4h_Bollinger_Breakout_ATRFilter_Volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for ATR calculation (more stable than lower timeframes)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate ATR(14) and ATR(50) on 1d timeframe
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    # ATR(14) and ATR(50)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    
    # Align ATR values to 4h timeframe
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    atr_50_aligned = align_htf_to_ltf(prices, df_1d, atr_50)
    
    # Bollinger Bands on 4h price data
    close_series = pd.Series(close)
    bb_middle = close_series.rolling(window=20, min_periods=20).mean().values
    bb_std = close_series.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    
    # Volume confirmation: >1.5x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50)  # BB and ATR50 warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_14_aligned[i]) or np.isnan(atr_50_aligned[i]) or 
            np.isnan(bb_middle[i]) or np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_atr14 = atr_14_aligned[i]
        curr_atr50 = atr_50_aligned[i]
        curr_bb_upper = bb_upper[i]
        curr_bb_lower = bb_lower[i]
        curr_bb_middle = bb_middle[i]
        curr_close = close[i]
        
        # Volatility filter: ATR(14) > 1.2 * ATR(50) ensures sufficient volatility
        vol_filter = curr_atr14 > 1.2 * curr_atr50
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price crosses below middle BB (mean reversion)
            if curr_close < curr_bb_middle:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above middle BB (mean reversion)
            if curr_close > curr_bb_middle:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long when price breaks above upper BB AND volatility filter AND volume confirmation
            if curr_close > curr_bb_upper and vol_filter and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below lower BB AND volatility filter AND volume confirmation
            elif curr_close < curr_bb_lower and vol_filter and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals