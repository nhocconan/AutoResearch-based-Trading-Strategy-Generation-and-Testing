# 1d_1w_keltner_breakout_volume_v1
# Hypothesis: Trade breakouts of weekly Keltner channels with volume confirmation on daily timeframe.
# Uses weekly ATR-based channels to capture medium-term trends, with volume surge to confirm breakout strength.
# Long when price breaks above upper channel with volume surge and weekly uptrend (price > weekly EMA50).
# Short when price breaks below lower channel with volume surge and weekly downtrend (price < weekly EMA50).
# Designed for 1d timeframe to target 7-25 trades/year (30-100 total over 4 years).
# Weekly trend filter ensures alignment with higher timeframe momentum, working in both bull and bear markets.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_keltner_breakout_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly data for Keltner channels
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate ATR(10) for weekly data
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # EMA50 for weekly trend
    ema50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Keltner channels: EMA20 ± 2*ATR
    ema20 = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    upper = ema20 + 2.0 * atr
    lower = ema20 - 2.0 * atr
    
    # Align weekly channels to daily timeframe
    upper_aligned = align_htf_to_ltf(prices, df_1w, upper)
    lower_aligned = align_htf_to_ltf(prices, df_1w, lower)
    ema50_aligned = align_htf_to_ltf(prices, df_1w, ema50)
    ema20_aligned = align_htf_to_ltf(prices, df_1w, ema20)
    
    # Volume confirmation: volume > 1.5x 20-day average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 50  # Ensure EMA50 is ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or 
            np.isnan(ema50_aligned[i]) or np.isnan(ema20_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Volume surge condition
        vol_surge = volume[i] > 1.5 * vol_ma_20[i] if vol_ma_20[i] > 0 else False
        
        if position == 1:  # Long position
            # Exit: price breaks below middle line (EMA20)
            if close[i] < ema20_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above middle line (EMA20)
            if close[i] > ema20_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price breaks above upper channel with volume surge and weekly uptrend
            if (close[i] > upper_aligned[i] and vol_surge and 
                close[i] > ema50_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below lower channel with volume surge and weekly downtrend
            elif (close[i] < lower_aligned[i] and vol_surge and 
                  close[i] < ema50_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals