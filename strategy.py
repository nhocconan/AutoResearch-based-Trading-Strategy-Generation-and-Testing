#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA trend with RSI mean reversion and choppiness regime filter
# Uses Kaufman Adaptive Moving Average (KAMA) for trend direction that adapts to market noise
# RSI(14) < 30 for long, > 70 for short in choppy markets (CHOP > 50) to catch reversals
# In trending markets (CHOP <= 50), follow KAMA direction to avoid whipsaw
# Designed to work in both bull and bear markets by adapting to regime
# Target: 30-100 trades over 4 years with discrete sizing 0.25

name = "1d_KAMA_RSI_ChopRegime_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate 1d KAMA (trend indicator)
    close_s = pd.Series(close)
    # Efficiency Ratio: |net change| / sum of absolute changes over 10 periods
    net_change = abs(close_s.diff(10))
    volatility = close_s.diff().abs().rolling(window=10).sum()
    er = net_change / volatility.replace(0, np.nan)
    # Smoothing constants: fastest SC=2/(2+1)=0.67, slowest SC=2/(30+1)=0.0645
    sc = (er * (0.67 - 0.0645) + 0.0645) ** 2
    # Handle NaN/inf
    sc = sc.fillna(0.0645)
    kama = close_s.copy()
    kama.iloc[0] = close_s.iloc[0]
    for i in range(1, len(close_s)):
        kama.iloc[i] = kama.iloc[i-1] + sc.iloc[i] * (close_s.iloc[i] - kama.iloc[i-1])
    kama_values = kama.values
    
    # Calculate RSI(14)
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.fillna(50).values  # neutral when undefined
    
    # Calculate Choppiness Index (CHOP) - measures if market is choppy (trending) or ranging
    # High CHOP (>61.8) = ranging/choppy, Low CHOP (<38.2) = trending
    tr1 = pd.Series(high).shift(1) - pd.Series(low).shift(1)
    tr2 = abs(pd.Series(high).shift(1) - pd.Series(close).shift(1))
    tr3 = abs(pd.Series(low).shift(1) - pd.Series(close).shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_sum = tr.rolling(window=14, min_periods=14).sum()
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min()
    chop = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(14)
    chop_values = chop.fillna(50).values  # neutral when undefined
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # start after warmup for indicators
        # Skip if any critical value is NaN
        if (np.isnan(kama_values[i]) or np.isnan(rsi_values[i]) or 
            np.isnan(chop_values[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # In choppy/ranging market (CHOP > 50): mean reversion at RSI extremes
            if chop_values[i] > 50:
                if rsi_values[i] < 30:  # oversold -> long
                    signals[i] = 0.25
                    position = 1
                elif rsi_values[i] > 70:  # overbought -> short
                    signals[i] = -0.25
                    position = -1
            # In trending market (CHOP <= 50): follow KAMA direction
            else:
                if close[i] > kama_values[i]:  # price above KAMA -> long
                    signals[i] = 0.25
                    position = 1
                elif close[i] < kama_values[i]:  # price below KAMA -> short
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: price crosses below KAMA OR RSI overbought in chop
            if close[i] < kama_values[i] or (chop_values[i] > 50 and rsi_values[i] > 70):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above KAMA OR RSI oversold in chop
            if close[i] > kama_values[i] or (chop_values[i] > 50 and rsi_values[i] < 30):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals