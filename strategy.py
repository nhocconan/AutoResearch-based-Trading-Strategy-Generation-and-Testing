#/usr/bin/env python3
"""
6h RSI with 1d Bollinger Band Filter v1
Hypothesis: RSI extremes (overbought/oversold) filtered by 1d Bollinger Band position (inside/outside bands) capture mean reversals in ranging markets and trend continuations in strong trends. The 1d Bollinger Band filter adapts to volatility regimes, avoiding trades during low-volatility squeezes and favoring breakouts during high volatility. This works in both bull and bear markets by dynamically adjusting to market conditions.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_rsi_1d_bb_filter_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    
    # 6s RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    gain_ma = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    loss_ma = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = gain_ma / (loss_ma + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # 1d Bollinger Bands
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ma_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper_bb = ma_20 + (2.0 * std_20)
    lower_bb = ma_20 - (2.0 * std_20)
    upper_bb_aligned = align_htf_to_ltf(prices, df_1d, upper_bb)
    lower_bb_aligned = align_htf_to_ltf(prices, df_1d, lower_bb)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(14, n):
        # Skip if any required data is NaN
        if (np.isnan(rsi[i]) or np.isnan(upper_bb_aligned[i]) or 
            np.isnan(lower_bb_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: RSI returns to neutral (40-60) or price touches opposite BB
            if rsi[i] >= 40 and rsi[i] <= 60:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: RSI returns to neutral (40-60) or price touches opposite BB
            if rsi[i] >= 40 and rsi[i] <= 60:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: RSI oversold (<30) and price below lower Bollinger Band (oversold extreme)
            if rsi[i] < 30 and close[i] < lower_bb_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Short: RSI overbought (>70) and price above upper Bollinger Band (overbought extreme)
            elif rsi[i] > 70 and close[i] > upper_bb_aligned[i]:
                position = -1
                signals[i] = -0.25
    
    return signals