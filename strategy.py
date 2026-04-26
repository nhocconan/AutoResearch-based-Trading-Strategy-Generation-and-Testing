#!/usr/bin/env python3
"""
4h_KAMA_Trend_Choppiness_Filter_v1
Hypothesis: 4h KAMA trend direction combined with choppiness regime filter and volume confirmation.
- Long when KAMA trend is up (price > KAMA) AND choppiness < 50 (trending market) AND volume > 1.3 * volume_ma(20)
- Short when KAMA trend is down (price < KAMA) AND choppiness < 50 AND volume > 1.3 * volume_ma(20)
- Uses KAMA from completed 4h bars for adaptive trend that reduces whipsaw in sideways markets
- Choppiness filter (CHOP < 50) ensures we only trade in trending conditions, avoiding range-bound false signals
- Volume confirmation ensures institutional participation and reduces false breakouts
- Designed for low frequency (target 15-35 trades/year) to minimize fee drag
- Exit on opposite KAMA crossover or choppiness > 60 (range market)
- Novelty: Combines adaptive trend (KAMA) with regime filter (choppiness) and volume for BTC/ETH edge in both bull/bear markets
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need enough data for calculations
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 4h data ONCE before loop for KAMA calculation
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate KAMA (Kaufman Adaptive Moving Average) from completed 4h bars
    # KAMA adapts its smoothing constant based on market efficiency ratio
    close_4h = pd.Series(df_4h['close'].values)
    # Efficiency Ratio: ER = |Price Change| / Sum of Absolute Price Changes
    change = abs(close_4h.diff(10))  # 10-period price change
    volatility = close_4h.diff().abs().rolling(window=10, min_periods=10).sum()  # 10-period volatility
    er = change / volatility.replace(0, np.nan)  # Avoid division by zero
    # Smoothing constants: fastest SC = 2/(2+1) = 0.667, slowest SC = 2/(30+1) = 0.0645
    sc = (er * (0.667 - 0.0645) + 0.0645) ** 2  # Square for smoother adaptation
    # Calculate KAMA
    kama = np.zeros_like(close_4h.values)
    kama[0] = close_4h.iloc[0]
    for i in range(1, len(close_4h)):
        if not np.isnan(sc.iloc[i]):
            kama[i] = kama[i-1] + sc.iloc[i] * (close_4h.iloc[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # Align KAMA to 4h timeframe (no additional delay needed for trend)
    kama_aligned = align_htf_to_ltf(prices, df_4h, kama)
    
    # Calculate choppiness index from completed 4h bars (regime filter)
    # Chop = 100 * log10(sum(TR) / (ATR * n)) / log10(n)
    tr1 = df_4h['high'] - df_4h['low']
    tr2 = abs(df_4h['high'] - df_4h['close'].shift(1))
    tr3 = abs(df_4h['low'] - df_4h['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean()
    chop = 100 * np.log10(tr.rolling(window=14, min_periods=14).sum() / (atr * 14)) / np.log10(14)
    chop_values = chop.values
    
    # Align choppiness to 4h timeframe (no additional delay needed)
    chop_aligned = align_htf_to_ltf(prices, df_4h, chop_values)
    
    # Calculate volume filter: volume > 1.3 * volume_ma(20) for confirmation
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.3 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 14 for chop, 10 for KAMA ER, 20 for volume MA)
    start_idx = max(20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(chop_aligned[i]) or
            np.isnan(volume_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # KAMA trend direction with choppiness and volume filter
        if position == 0:
            # Long: Price above KAMA AND chop < 50 (trending) AND volume spike
            if close[i] > kama_aligned[i] and chop_aligned[i] < 50 and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price below KAMA AND chop < 50 (trending) AND volume spike
            elif close[i] < kama_aligned[i] and chop_aligned[i] < 50 and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: Price falls below KAMA OR chop > 60 (range) OR volume drops
            if close[i] < kama_aligned[i] or chop_aligned[i] > 60:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: Price rises above KAMA OR chop > 60 (range) OR volume drops
            if close[i] > kama_aligned[i] or chop_aligned[i] > 60:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_KAMA_Trend_Choppiness_Filter_v1"
timeframe = "4h"
leverage = 1.0