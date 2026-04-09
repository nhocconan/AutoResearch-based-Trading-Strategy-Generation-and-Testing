#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Bollinger Bands squeeze + 1w volume confirmation
# Bollinger Bands squeeze (band width < 20th percentile) indicates low volatility and impending breakout
# In squeeze: wait for breakout above upper band (long) or below lower band (short) with volume confirmation
# Outside squeeze: no trades to avoid whipsaw in ranging markets
# Uses 1w volume ratio > 1.5 to confirm institutional participation
# Works in bull/bear markets: captures explosive moves after consolidation periods

name = "6h_1d_1w_bb_squeeze_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d Bollinger Bands (20, 2)
    close_s_1d = pd.Series(close_1d)
    basis_1d = close_s_1d.rolling(window=20, min_periods=20).mean()
    dev_1d = close_s_1d.rolling(window=20, min_periods=20).std()
    upper_1d = basis_1d + 2.0 * dev_1d
    lower_1d = basis_1d - 2.0 * dev_1d
    
    # Calculate Bollinger Band Width
    bb_width_1d = (upper_1d - lower_1d) / basis_1d
    
    # Calculate 20th percentile of BB width for squeeze detection (using expanding window)
    bb_width_percentile_1d = pd.Series(bb_width_1d).expanding(min_periods=50).quantile(0.20).values
    
    # Squeeze condition: BB width < 20th percentile
    squeeze_1d = bb_width_1d < bb_width_percentile_1d
    
    # Breakout conditions
    breakout_up_1d = close_1d > upper_1d
    breakout_down_1d = close_1d < lower_1d
    
    # Load 1w data for volume confirmation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    volume_1w = df_1w['volume'].values
    
    # Calculate 1w volume ratio (current vs 20-period average)
    volume_s_1w = pd.Series(volume_1w)
    volume_ma_1w = volume_s_1w.rolling(window=20, min_periods=20).mean()
    volume_ratio_1w = volume_1w / volume_ma_1w
    
    # Align 1d indicators to 6h timeframe
    squeeze_1d_aligned = align_htf_to_ltf(prices, df_1d, squeeze_1d.astype(float))
    breakout_up_1d_aligned = align_htf_to_ltf(prices, df_1d, breakout_up_1d.astype(float))
    breakout_down_1d_aligned = align_htf_to_ltf(prices, df_1d, breakout_down_1d.astype(float))
    
    # Align 1w volume ratio to 6h timeframe
    volume_ratio_1w_aligned = align_htf_to_ltf(prices, df_1w, volume_ratio_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(squeeze_1d_aligned[i]) or np.isnan(breakout_up_1d_aligned[i]) or
            np.isnan(breakout_down_1d_aligned[i]) or np.isnan(volume_ratio_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit long if price returns to basis (mean reversion) or stop loss
            if close[i] <= basis_1d.iloc[-1] if not np.isnan(basis_1d.iloc[-1]) else close[i] <= close[i-1]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit short if price returns to basis (mean reversion) or stop loss
            if close[i] >= basis_1d.iloc[-1] if not np.isnan(basis_1d.iloc[-1]) else close[i] >= close[i-1]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for breakout from Bollinger Bands squeeze with volume confirmation
            if (squeeze_1d_aligned[i] > 0.5 and  # In squeeze
                volume_ratio_1w_aligned[i] > 1.5):  # Volume confirmation
                
                if breakout_up_1d_aligned[i] > 0.5:  # Breakout above upper band
                    position = 1
                    signals[i] = 0.25
                elif breakout_down_1d_aligned[i] > 0.5:  # Breakout below lower band
                    position = -1
                    signals[i] = -0.25
    
    return signals