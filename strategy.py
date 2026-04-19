#/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Weekly donchian breakout with daily volume confirmation and ATR filter
# Weekly donchian captures long-term structure (resistance/support breaks)
# Daily volume > 1.5x average confirms institutional participation
# ATR-based stop (2x ATR) limits drawdown in volatile moves
# Target: 30-100 total trades over 4 years (7-25/year) with disciplined entries
# Works in bull/bear: breaks above weekly high in bull, below weekly low in bear
name = "1d_WeeklyDonchian_Volume_ATR"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly Donchian channels (20-period)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly high/low over 20 periods
    weekly_high = pd.Series(df_1w['high']).rolling(window=20, min_periods=20).max().values
    weekly_low = pd.Series(df_1w['low']).rolling(window=20, min_periods=20).min().values
    
    # Align to daily timeframe
    weekly_high_aligned = align_htf_to_ltf(prices, df_1w, weekly_high)
    weekly_low_aligned = align_htf_to_ltf(prices, df_1w, weekly_low)
    
    # Daily volume confirmation: > 1.5x 20-day average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 1.5)
    
    # ATR for stop calculation (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0.0
    tr2[0] = 0.0
    tr3[0] = 0.0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr = np.zeros(n)
    atr[13] = tr[:14].mean()
    for i in range(14, n):
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 14)  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(weekly_high_aligned[i]) or np.isnan(weekly_low_aligned[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Price breaks above weekly high + volume confirmation
            if close[i] > weekly_high_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below weekly low + volume confirmation
            elif close[i] < weekly_low_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks below weekly low OR 2x ATR trailing stop
            if (close[i] < weekly_low_aligned[i]) or (close[i] < np.max(high[max(0, i-5):i+1]) - 2 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price breaks above weekly high OR 2x ATR trailing stop
            if (close[i] > weekly_high_aligned[i]) or (close[i] > np.min(low[max(0, i-5):i+1]) + 2 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals