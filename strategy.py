#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA trend direction with RSI(14) momentum and choppiness regime filter
# - Uses 1w EMA200 as higher timeframe trend filter to avoid counter-trend trades
# - Long when KAMA(14,2,30) rising AND RSI(14) > 50 AND Choppiness Index(14) < 61.8 (trending regime)
# - Short when KAMA falling AND RSI(14) < 50 AND Choppiness Index(14) < 61.8
# - Exit when opposite signal occurs or Choppiness Index > 61.8 (range regime)
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Target: 15-25 trades/year on 1d timeframe (60-100 total over 4 years)
# - KAMA adapts to market noise, RSI confirms momentum, Choppiness filters ranging markets

name = "1d_1w_kama_rsi_chop_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Pre-compute 1w EMA(200) for trend filter
    close_1w = df_1w['close'].values
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # Pre-compute KAMA(14,2,30) - Kaufman Adaptive Moving Average
    close = prices['close'].values
    direction = np.abs(np.diff(close, n=10))  # 10-period net change
    volatility = np.sum(np.abs(np.diff(close, n=1)), axis=0)  # 10-period sum of absolute changes
    # Avoid division by zero
    volatility = np.where(volatility == 0, 1e-10, volatility)
    er = direction / volatility  # Efficiency Ratio
    # Smooth ER
    er = pd.Series(er).ewm(alpha=1, adjust=False).fillna(0).values
    # Constants for fast and slow EMA
    fast_sc = 2 / (2 + 1)  # 2-period EMA
    slow_sc = 2 / (30 + 1)  # 30-period EMA
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2  # Smoothing Constant
    # Calculate KAMA
    kama = np.full_like(close, np.nan)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Pre-compute RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss == 0, 100, avg_gain / avg_loss)
    rsi = 100 - (100 / (1 + rs))
    
    # Pre-compute Choppiness Index(14)
    high = prices['high'].values
    low = prices['low'].values
    atr_1 = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    atr_1[0] = high[0] - low[0]  # first value
    atr_sum = pd.Series(atr_1).rolling(window=14, min_periods=14).sum().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(14)
    # Handle division by zero and invalid values
    chop = np.where((highest_high - lowest_low) == 0, 100, chop)
    chop = np.where(np.isnan(chop), 100, chop)
    
    # Align HTF EMA200 to LTF
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or 
            np.isnan(ema200_1w_aligned[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long when KAMA rising AND RSI > 50 AND trending regime (CHOP < 61.8)
            if (kama[i] > kama[i-1] and 
                rsi[i] > 50 and 
                chop[i] < 61.8 and
                close[i] > ema200_1w_aligned[i]):  # price above 1w EMA200 for long bias
                position = 1
                signals[i] = 0.25
            # Short when KAMA falling AND RSI < 50 AND trending regime (CHOP < 61.8)
            elif (kama[i] < kama[i-1] and 
                  rsi[i] < 50 and 
                  chop[i] < 61.8 and
                  close[i] < ema200_1w_aligned[i]):  # price below 1w EMA200 for short bias
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit when opposite KAMA direction OR chop > 61.8 (range regime) OR price crosses 1w EMA200
            exit_signal = False
            if position == 1:  # Long position
                if (kama[i] < kama[i-1] or  # KAMA falling
                    chop[i] > 61.8 or       # ranging regime
                    close[i] < ema200_1w_aligned[i]):  # price below 1w EMA200
                    exit_signal = True
            elif position == -1:  # Short position
                if (kama[i] > kama[i-1] or  # KAMA rising
                    chop[i] > 61.8 or       # ranging regime
                    close[i] > ema200_1w_aligned[i]):  # price above 1w EMA200
                    exit_signal = True
            
            if exit_signal:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals