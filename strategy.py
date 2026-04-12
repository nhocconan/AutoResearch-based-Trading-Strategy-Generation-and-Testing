#/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h_1d_adaptive_kama_breakout_v1
# Adaptive KAMA direction filter (2-period) + price breaking above/below ATR-based channel.
# Uses KAMA to determine trend direction (bullish if KAMA rising >2 periods, bearish if falling >2 periods).
# Entry only when price breaks above upper channel (ATR*1.5) in bull trend or below lower channel in bear trend.
# Includes volume confirmation (>1.5x 20-period average) and chop filter (CHOP < 61.8) to avoid false signals.
# Target: 20-35 trades/year per symbol with strong trend capture in both bull and bear markets.
name = "4h_1d_adaptive_kama_breakout_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for adaptive filtering
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1-day KAMA for trend direction
    close_1d = df_1d['close'].values
    # Efficiency Ratio
    change = np.abs(np.diff(close_1d))
    volatility = np.sum(np.abs(np.diff(close_1d, 1)), axis=0) if len(close_1d) > 1 else np.array([0])
    # Simplified ER calculation for array
    er = np.zeros_like(close_1d)
    for i in range(10, len(close_1d)):
        if i >= 10:
            change_val = np.abs(close_1d[i] - close_1d[i-9])
            volatility_val = np.sum(np.abs(np.diff(close_1d[i-9:i+1])))
            if volatility_val > 0:
                er[i] = change_val / volatility_val
            else:
                er[i] = 1.0
    # Smoothing constants
    sc = (er * 0.2889 + 0.0645) ** 2  # 2/(2+1) = 0.6667, 2/(30+1)=0.0645
    # Calculate KAMA
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # Determine trend direction: rising if KAMA up >2 periods, falling if down >2 periods
    kama_diff = np.diff(kama, prepend=kama[0])
    kama_rising = kama_diff > 0
    kama_falling = kama_diff < 0
    # Count consecutive periods
    rising_count = np.zeros_like(kama)
    falling_count = np.zeros_like(kama)
    for i in range(1, len(kama)):
        if kama_rising[i]:
            rising_count[i] = rising_count[i-1] + 1
        else:
            rising_count[i] = 0
        if kama_falling[i]:
            falling_count[i] = falling_count[i-1] + 1
        else:
            falling_count[i] = 0
    kama_bull = rising_count >= 2
    kama_bear = falling_count >= 2
    
    # Align KAMA trend to 4h timeframe
    kama_bull_aligned = align_htf_to_ltf(prices, df_1d, kama_bull.astype(float))
    kama_bear_aligned = align_htf_to_ltf(prices, df_1d, kama_bear.astype(float))
    
    # Calculate ATR-based channel for breakout
    atr_period = 14
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    # Upper and lower channels (ATR * 1.5)
    upper_channel = close + atr * 1.5
    lower_channel = close - atr * 1.5
    
    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    # Chop regime filter: avoid choppy markets (CHOP > 61.8)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10((highest_high - lowest_low) / (atr * np.sqrt(14))) / np.log10(14)
    chop_filter = chop < 61.8  # trending market
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # start after warmup
        # Skip if data not ready
        if np.isnan(kama_bull_aligned[i]) or np.isnan(kama_bear_aligned[i]) or np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]):
            signals[i] = 0.0
            continue
        
        # Check volume and chop filters
        if not (vol_confirm[i] and chop_filter[i]):
            # Hold current position if filters fail
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Long signal: KAMA bull trend + price breaks above upper channel
        if kama_bull_aligned[i] > 0.5 and close[i] > upper_channel[i] and position != 1:
            position = 1
            signals[i] = 0.25
        # Short signal: KAMA bear trend + price breaks below lower channel
        elif kama_bear_aligned[i] > 0.5 and close[i] < lower_channel[i] and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit conditions: opposite KAMA signal or channel re-entry
        elif (kama_bear_aligned[i] > 0.5 and position == 1) or (kama_bull_aligned[i] > 0.5 and position == -1):
            position = 0
            signals[i] = 0.0
        elif close[i] < lower_channel[i] and position == 1:
            position = 0
            signals[i] = 0.0
        elif close[i] > upper_channel[i] and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals