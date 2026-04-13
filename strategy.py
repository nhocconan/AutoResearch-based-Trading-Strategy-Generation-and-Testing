# 6h_1d_Camarilla_Pivot_Breakout_With_Volume_Filter_v1
# Hypothesis: Camarilla pivot levels from 1d act as strong support/resistance. Breakout above R4 or below S4 with volume > 1.5x 20-period average triggers continuation. Uses 60% position size to manage risk. Works in bull via R4 breakouts and bear via S4 breakdowns.
# Target: 20-50 trades/year per symbol.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate ATR for volatility filter
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[0], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_expansion = volume > (vol_ma_20 * 1.5)
    
    # Get 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        camarilla_r4 = np.full(n, np.nan)
        camarilla_s4 = np.full(n, np.nan)
    else:
        # Previous day's OHLC
        prev_close = df_1d['close'].shift(1).values
        prev_high = df_1d['high'].shift(1).values
        prev_low = df_1d['low'].shift(1).values
        
        # Camarilla calculations
        camarilla_r4 = prev_close + (prev_high - prev_low) * 1.1 / 2
        camarilla_s4 = prev_close - (prev_high - prev_low) * 1.1 / 2
        
        # Align to 6h timeframe (wait for 1d bar to close)
        camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
        camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.60  # 60% position size
    
    for i in range(30, n):
        # Skip if any required data is not ready
        if (np.isnan(camarilla_r4_aligned[i]) if len(df_1d) >= 2 else True or
            np.isnan(camarilla_s4_aligned[i]) if len(df_1d) >= 2 else True or
            np.isnan(volume_expansion[i])):
            signals[i] = 0.0
            continue
        
        # Long signal: break above R4 with volume expansion
        long_signal = (high[i] > camarilla_r4_aligned[i] and volume_expansion[i])
        
        # Short signal: break below S4 with volume expansion
        short_signal = (low[i] < camarilla_s4_aligned[i] and volume_expansion[i])
        
        if long_signal and position != 1:
            position = 1
            signals[i] = position_size
        elif short_signal and position != -1:
            position = -1
            signals[i] = -position_size
        else:
            # Hold current position
            signals[i] = position_size if position == 1 else (-position_size if position == -1 else 0.0)
    
    return signals

name = "6h_1d_Camarilla_Pivot_Breakout_With_Volume_Filter_v1"
timeframe = "6h"
leverage = 1.0