#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA trend with 1w trend filter and volume confirmation.
# Long when KAMA shows upward trend on 1d during bullish week with volume > 1.2x 20-period average.
# Short when KAMA shows downward trend on 1d during bearish week with volume confirmation.
# Weekly trend filter avoids counter-trend trades. KAMA adapts to volatility for better trend detection.
# Target: 50-100 total trades over 4 years (12-25/year) to stay within optimal range.

name = "1d_kama_1w_trend_vol_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price and volume data
    close = prices['close'].values
    volume = prices['volume'].values
    
    # KAMA (Kaufman Adaptive Moving Average) - 14-period
    # ER = Efficiency Ratio, SC = Smoothing Constant
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close, prepend=close[0])), axis=0) if False else None  # placeholder
    
    # Calculate ER properly
    diff = np.diff(close, prepend=close[0])
    abs_diff = np.abs(diff)
    change_10 = np.abs(np.diff(close, n=10, prepend=close[:10])) if len(close) >= 10 else np.full(len(close), np.nan)
    vol_10 = np.sum(np.abs(np.diff(close, n=1, prepend=close[0])), axis=0) if False else None  # placeholder
    
    # Simpler approach: use pandas for rolling calculations
    close_s = pd.Series(close)
    change = np.abs(close_s.diff(1))
    volatility = close_s.diff(1).abs().rolling(window=10, min_periods=10).sum()
    er = change / volatility.replace(0, np.nan)
    er = er.fillna(0)
    
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        if np.isnan(sc[i-1]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i-1] * (close[i] - kama[i-1])
    
    # KAMA direction: slope > 0 for up, < 0 for down
    kama_series = pd.Series(kama)
    kama_slope = kama_series.diff(2)  # 2-period slope
    
    # Weekly trend filter
    df_1w = get_htf_data(prices, '1w')
    weekly_open = df_1w['open'].values
    weekly_close = df_1w['close'].values
    weekly_bullish = weekly_close > weekly_open
    weekly_bearish = weekly_close < weekly_open
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish)
    weekly_bearish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bearish)
    
    # Volume filter
    volume_series = pd.Series(volume)
    vol_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if weekly trend data not available
        if np.isnan(weekly_bullish_aligned[i]) or np.isnan(weekly_bearish_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.2
        
        # Check exits
        if position == 1:  # long position
            # Exit: KAMA turns down or weekly turn bearish
            if (kama_slope[i] < 0 or 
                weekly_bearish_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: KAMA turns up or weekly turn bullish
            if (kama_slope[i] > 0 or 
                weekly_bullish_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation and weekly trend filter
            if volume_filter:
                # Long: KAMA trending up during bullish week
                if (kama_slope[i] > 0 and 
                    weekly_bullish_aligned[i]):
                    signals[i] = 0.25
                    position = 1
                # Short: KAMA trending down during bearish week
                elif (kama_slope[i] < 0 and 
                      weekly_bearish_aligned[i]):
                    signals[i] = -0.25
                    position = -1
    
    return signals