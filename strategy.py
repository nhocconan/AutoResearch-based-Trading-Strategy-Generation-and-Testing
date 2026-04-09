#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA trend with 1w volume confirmation and chop regime filter
# - Uses Kaufman Adaptive Moving Average (KAMA) on 1d to determine trend direction
# - Requires 1w volume > 1.5 * 20-period volume average for confirmation (reduces false signals)
# - Uses Choppiness Index (CHOP) on 1d to filter ranging markets (CHOP > 61.8 = range, avoid trend signals)
# - Long when: price > KAMA(1d) AND volume_confirm AND CHOP <= 61.8 (trending)
# - Short when: price < KAMA(1d) AND volume_confirm AND CHOP <= 61.8 (trending)
# - ATR-based stoploss (2.5 * ATR) to manage risk
# - Position size: 0.25 (25% of capital) to balance return and drawdown
# - Designed to work in both bull and bear markets by following the adaptive trend
# - Target: 15-30 trades/year on 1d timeframe (60-120 total over 4 years) to minimize fee drag

name = "1d_1w_kama_volume_chop_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Pre-compute 1w volume confirmation: volume > 1.5 * 20-period average
    volume_1w = df_1w['volume'].values
    vol_ma_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    volume_confirm_1w = volume_1w > (1.5 * vol_ma_1w)
    volume_confirm_1w_aligned = align_htf_to_ltf(prices, df_1w, volume_confirm_1w)
    
    # Pre-compute 1d KAMA (trend indicator)
    close = prices['close'].values
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(close - np.roll(close, 10))
    volatility = np.sum(np.abs(np.diff(close, prepend=close[0])), axis=0)  # approximate for vectorized
    # Correct volatility calculation: sum of absolute changes over 10 periods
    volatility = np.zeros_like(close)
    for i in range(10, len(close)):
        volatility[i] = np.sum(np.abs(np.diff(close[i-9:i+1])))
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    # Initialize KAMA
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # start after 10 periods
    for i in range(10, n):
        if not np.isnan(kama[i-1]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = close[i]
    
    # Pre-compute 1d Choppiness Index (regime filter)
    high = prices['high'].values
    low = prices['low'].values
    # True Range over 14 periods
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    # Sum of TR over 14 periods
    sum_tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    # MaxHigh - MinLow over 14 periods
    max_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    range_14 = max_high_14 - min_low_14
    # Choppiness Index: CHOP = 100 * log10(sum_tr_14 / range_14) / log10(14)
    chop = np.where(range_14 > 0, 100 * np.log10(sum_tr_14 / range_14) / np.log10(14), 50)
    chop_filter = chop <= 61.8  # trending market
    
    # Pre-compute 1d ATR for stoploss
    atr = atr_14  # already computed above
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    highest_high_since_entry = 0.0
    lowest_low_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(kama[i]) or np.isnan(volume_confirm_1w_aligned[i]) or
            np.isnan(chop_filter[i]) or np.isnan(atr[i]) or atr[i] <= 0):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Update highest high since entry
            highest_high_since_entry = max(highest_high_since_entry, high[i])
            
            # Exit conditions: stoploss or trend reversal
            if close[i] < highest_high_since_entry - 2.5 * atr[i]:  # ATR stop
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
                signals[i] = 0.0
            elif close[i] < kama[i]:  # trend reversal (price below KAMA)
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest low since entry
            lowest_low_since_entry = min(lowest_low_since_entry, low[i])
            
            # Exit conditions: stoploss or trend reversal
            if close[i] > lowest_low_since_entry + 2.5 * atr[i]:  # ATR stop
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
                signals[i] = 0.0
            elif close[i] > kama[i]:  # trend reversal (price above KAMA)
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for trend entries with volume confirmation and trending regime
            if close[i] > kama[i] and volume_confirm_1w_aligned[i] and chop_filter[i]:
                position = 1
                highest_high_since_entry = high[i]
                lowest_low_since_entry = low[i]
                signals[i] = 0.25
            elif close[i] < kama[i] and volume_confirm_1w_aligned[i] and chop_filter[i]:
                position = -1
                highest_high_since_entry = high[i]
                lowest_low_since_entry = low[i]
                signals[i] = -0.25
    
    return signals