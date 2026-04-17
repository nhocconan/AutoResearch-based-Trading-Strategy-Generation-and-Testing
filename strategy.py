# 1d_Weekly_Keltner_MeanReversion
# Hypothesis: On daily timeframe, mean-revert at Keltner Channel bands (ATR-based) with weekly trend filter.
# Enter long when price touches lower band in weekly uptrend, short when touches upper band in weekly downtrend.
# Uses volume confirmation to avoid false signals. Designed for 10-25 trades/year to minimize fee drag.
# Works in bull/bear via weekly trend alignment: only trade in direction of weekly trend.

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
    
    # === Daily data for Keltner Channel ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Keltner Channel: EMA(20) ± ATR(10) * 2
    ema20_1d = pd.Series(close_1d).ewm(span=20, min_periods=20, adjust=False).mean().values
    atr10_1d = pd.Series(high_1d - low_1d).rolling(window=10, min_periods=10).mean().values
    upper_keltner = ema20_1d + 2 * atr10_1d
    lower_keltner = ema20_1d - 2 * atr10_1d
    
    # Align Keltner bands to daily timeframe (already aligned, but explicit)
    upper_keltner_aligned = align_htf_to_ltf(prices, df_1d, upper_keltner)
    lower_keltner_aligned = align_htf_to_ltf(prices, df_1d, lower_keltner)
    ema20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema20_1d)
    
    # === Weekly trend filter ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    # Weekly EMA34 for trend
    ema34_1w = pd.Series(close_1w).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # === Daily volume average (20-period) for confirmation ===
    volume_1d = df_1d['volume'].values
    vol_avg20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg20_1d)
    
    signals = np.zeros(n)
    
    # Warmup: covers EMA20, ATR10, EMA34 weekly, volume average
    warmup = max(20, 10, 34, 20)
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(upper_keltner_aligned[i]) or 
            np.isnan(lower_keltner_aligned[i]) or 
            np.isnan(ema34_1w_aligned[i]) or 
            np.isnan(vol_avg20_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Get current daily volume
        vol_1d_current = align_htf_to_ltf(prices, df_1d, volume_1d)[i]
        
        # Volume filter: current volume > 1.3x 20-period average
        vol_filter = vol_1d_current > 1.3 * vol_avg20_1d_aligned[i]
        
        # Entry conditions
        if position == 0:
            # Long: price touches lower Keltner band + weekly uptrend + volume
            if close[i] <= lower_keltner_aligned[i] and close[i] > ema34_1w_aligned[i] and vol_filter:
                signals[i] = 0.25
                position = 1
                continue
            # Short: price touches upper Keltner band + weekly downtrend + volume
            elif close[i] >= upper_keltner_aligned[i] and close[i] < ema34_1w_aligned[i] and vol_filter:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit conditions: return to EMA20 (mean reversion target)
        elif position == 1:
            if close[i] >= ema20_1d_aligned[i]:  # exit long when price returns to EMA20
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            if close[i] <= ema20_1d_aligned[i]:  # exit short when price returns to EMA20
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Weekly_Keltner_MeanReversion"
timeframe = "1d"
leverage = 1.0