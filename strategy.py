#!/usr/bin/env python3
# Hypothesis: 1d KAMA trend with RSI(14) mean reversion and chop filter (CHOP > 61.8)
# Long when: KAMA rising, RSI < 30, choppy market (CHOP > 61.8)
# Short when: KAMA falling, RSI > 70, choppy market (CHOP > 61.8)
# Exit when: RSI crosses 50 or KAMA trend reverses
# Position size: 0.25 to limit drawdown. Target: 10-25 trades/year.
# Designed to work in ranging markets (chop) with mean reversion, avoids strong trends.

name = "1d_KAMA_RSI_Chop_MeanReversion"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # KAMA calculation
    close_series = pd.Series(close)
    direction = abs(close_series - close_series.shift(10))
    volatility = close_series.diff().abs().rolling(window=10).sum()
    er = direction / volatility.replace(0, np.nan)
    sc = (er * (0.6645 - 0.0645) + 0.0645) ** 2
    kama = close_series.copy()
    for i in range(1, len(kama)):
        if not np.isnan(sc.iloc[i]):
            kama.iloc[i] = kama.iloc[i-1] + sc.iloc[i] * (close_series.iloc[i] - kama.iloc[i-1])
        else:
            kama.iloc[i] = kama.iloc[i-1]
    kama_values = kama.values
    kama_prev = np.roll(kama_values, 1)
    kama_prev[0] = kama_values[0]
    kama_rising = kama_values > kama_prev
    kama_falling = kama_values < kama_prev
    
    # RSI(14)
    delta = close_series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Choppiness Index (CHOP)
    atr = pd.Series(np.maximum.reduce([
        high - low,
        np.abs(high - close_series.shift(1)),
        np.abs(low - close_series.shift(1))
    ]))
    atr_sum = atr.rolling(window=14, min_periods=14).sum()
    highest_high = close_series.rolling(window=14, min_periods=14).max()
    lowest_low = close_series.rolling(window=14, min_periods=14).min()
    chop = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(14)
    chop_values = chop.values
    
    # Get 1w data for trend filter (optional, not used in entry but can be added)
    # df_1w = get_htf_data(prices, '1w')
    # if len(df_1w) >= 10:
    #     ema_10_1w = pd.Series(df_1w['close']).ewm(span=10, adjust=False, min_periods=10).mean().values
    #     ema_10_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_10_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(kama_values[i]) or np.isnan(rsi_values[i]) or 
            np.isnan(chop_values[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: KAMA rising, RSI < 30, choppy market
            if (kama_rising[i] and rsi_values[i] < 30 and chop_values[i] > 61.8):
                signals[i] = 0.25
                position = 1
            # Enter short: KAMA falling, RSI > 70, choppy market
            elif (kama_falling[i] and rsi_values[i] > 70 and chop_values[i] > 61.8):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: RSI crosses above 50 OR KAMA turns down
            if (rsi_values[i] > 50) or (not kama_rising[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: RSI crosses below 50 OR KAMA turns up
            if (rsi_values[i] < 50) or (not kama_falling[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals