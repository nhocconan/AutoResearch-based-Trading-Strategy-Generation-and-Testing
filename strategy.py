#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA trend with RSI(14) mean reversion entries and chop regime filter
# Uses 1d Kaufman Adaptive Moving Average (KAMA) for trend direction (ER=10, FAST=2, SLOW=30)
# Enters long when RSI < 30 (oversold) in uptrend, short when RSI > 70 (overbought) in downtrend
# Uses 1d choppiness index (CHOP) > 61.8 for range regime (mean reversion) and < 38.2 for trending
# Volume confirmation: current volume > 1.5x 20-bar average to ensure participation
# ATR-based trailing stop via signal=0 when price retraces 25% of ATR from extreme
# Discrete sizing 0.25 to balance profit potential and fee drag; target 50-80 total trades over 4 years (12-20/year)
# Works in both bull/bear: KAMA adapts to trend changes, RSI mean reversion captures swings, volume filter ensures participation

name = "1d_KAMA_RSI_Chop_Volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 50:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA50 trend filter for higher timeframe bias
    close_1w_series = pd.Series(close_1w)
    ema50_1w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 1d KAMA (ER=10, FAST=2, SLOW=30)
    close_series = pd.Series(close)
    change = np.abs(close_series.diff(10).values)  # 10-period net change
    volatility = np.abs(close_series.diff(1)).rolling(window=10, min_periods=10).sum().values  # 10-period volatility
    er = np.where(volatility != 0, change / volatility, 0)  # Efficiency Ratio
    sc = (er * (2/2 - 2/30) + 2/30) ** 2  # Smoothing Constant
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # Seed value
    for i in range(10, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate 1d RSI(14)
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Calculate 1d Choppiness Index (CHOP)
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(14)
    
    # Calculate ATR(14) for stoploss
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate volume confirmation (>1.5x 20-bar average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    # Align HTF indicators to 1d timeframe
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    long_extreme = 0.0
    short_extreme = 0.0
    
    for i in range(50, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(kama[i]) or np.isnan(rsi_values[i]) or np.isnan(chop[i]) or 
            np.isnan(atr[i]) or np.isnan(volume_filter[i]) or np.isnan(ema50_1w_aligned[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
                long_extreme = 0.0
                short_extreme = 0.0
            continue
        
        if position == 0:
            # Long entry: RSI < 30 (oversold) AND price > KAMA (uptrend) AND CHOP > 61.8 (range) AND volume confirmation
            if rsi_values[i] < 30 and close[i] > kama[i] and chop[i] > 61.8 and volume_filter[i]:
                signals[i] = 0.25
                position = 1
                long_extreme = close[i]
            # Short entry: RSI > 70 (overbought) AND price < KAMA (downtrend) AND CHOP > 61.8 (range) AND volume confirmation
            elif rsi_values[i] > 70 and close[i] < kama[i] and chop[i] > 61.8 and volume_filter[i]:
                signals[i] = -0.25
                position = -1
                short_extreme = close[i]
        elif position == 1:
            # Update long extreme
            long_extreme = max(long_extreme, close[i])
            # Exit long: price retraces 25% of ATR from extreme OR RSI > 50 (mean reversion complete)
            if close[i] <= long_extreme - 0.25 * atr[i] or rsi_values[i] > 50:
                signals[i] = 0.0
                position = 0
                long_extreme = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Update short extreme
            short_extreme = min(short_extreme, close[i])
            # Exit short: price retraces 25% of ATR from extreme OR RSI < 50 (mean reversion complete)
            if close[i] >= short_extreme + 0.25 * atr[i] or rsi_values[i] < 50:
                signals[i] = 0.0
                position = 0
                short_extreme = 0.0
            else:
                signals[i] = -0.25
    
    return signals