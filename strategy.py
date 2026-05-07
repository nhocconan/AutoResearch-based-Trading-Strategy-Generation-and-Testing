# 4h_Camarilla_R1S1_Breakout_1dTrend_Volume_Momentum
# Hypothesis: Price breaking above Camarilla R1 or below S1 levels from the prior day,
# combined with 1d EMA50 trend filter and volume confirmation, captures momentum in both bull and bear markets.
# The 1d EMA50 ensures alignment with higher timeframe momentum, reducing false breakouts.
# Momentum filter (RSI > 50 for long, < 50 for short) adds confirmation.
# Target: 20-50 trades/year on 4h to avoid fee drag.
name = "4h_Camarilla_R1S1_Breakout_1dTrend_Volume_Momentum"
timeframe = "4h"
leverage = 1.0

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
    
    # Get 1d data for Camarilla levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d Camarilla R1/S1 levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    range_1d = high_1d - low_1d
    r1_1d = close_1d + 1.1666 * range_1d * 0.5 / 2  # R1 = C + 1.1666*(H-L)*0.5/2
    s1_1d = close_1d - 1.1666 * range_1d * 0.5 / 2  # S1 = C - 1.1666*(H-L)*0.5/2
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Momentum: RSI(14)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # Align all to 4h timeframe
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume filter: current volume > 1.3 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0
    
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_avg[i]) or np.isnan(rsi[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        bars_since_entry += 1
        
        if position == 0:
            # Long: price breaks above R1 + 1d uptrend + volume + RSI > 50
            if close[i] > r1_1d_aligned[i] and close[i] > ema_50_1d_aligned[i] and volume_filter[i] and rsi[i] > 50:
                signals[i] = 0.25
                position = 1
                bars_since_entry = 0
            # Short: price breaks below S1 + 1d downtrend + volume + RSI < 50
            elif close[i] < s1_1d_aligned[i] and close[i] < ema_50_1d_aligned[i] and volume_filter[i] and rsi[i] < 50:
                signals[i] = -0.25
                position = -1
                bars_since_entry = 0
        elif position != 0:
            # Minimum holding period of 2 bars to reduce trade frequency
            if bars_since_entry < 2:
                signals[i] = 0.25 if position == 1 else -0.25
                continue
            
            # Exit: price crosses back through the opposite S1/R1 level
            if position == 1:
                if close[i] < s1_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > r1_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.25
    
    return signals