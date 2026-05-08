#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_KAMA_RSI_Chop_Trend_v2"
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
    
    # Get weekly data once
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # KAMA calculation
    close_series = pd.Series(close)
    change = abs(close_series.diff(1))
    volatility = change.rolling(window=10, min_periods=10).sum()
    er = change / volatility.replace(0, np.nan)
    er = er.fillna(0)
    sc = (er * (2/2 - 2/30) + 2/30) ** 2
    kama = pd.Series(index=close_series.index, dtype=float)
    kama.iloc[0] = close.iloc[0]
    for i in range(1, len(close_series)):
        kama.iloc[i] = kama.iloc[i-1] + sc.iloc[i] * (close_series.iloc[i] - kama.iloc[i-1])
    kama_values = kama.values
    
    # RSI(14)
    delta = close_series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.fillna(50).values
    
    # Choppiness Index (14)
    atr = pd.Series(np.sqrt((high - low)**2)).rolling(window=14, min_periods=14).mean()
    high_low = pd.Series(high).rolling(window=14, min_periods=14).max()
    low_low = pd.Series(low).rolling(window=14, min_periods=14).min()
    chop = 100 * np.log10(atr.sum() / (high_low - low_low)) / np.log10(14)
    chop_values = chop.values
    
    # Weekly EMA34 trend filter
    close_1w = df_1w['close'].values
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_1w = (close_1w > ema34_1w).astype(float)
    trend_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_1w)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 30  # warmup for all indicators
    
    for i in range(start_idx, n):
        if np.isnan(kama_values[i]) or np.isnan(rsi_values[i]) or np.isnan(chop_values[i]) or np.isnan(trend_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price > KAMA, RSI > 50, chop < 61.8 (trending), weekly uptrend
            long_cond = (close[i] > kama_values[i] and 
                         rsi_values[i] > 50 and 
                         chop_values[i] < 61.8 and 
                         trend_1w_aligned[i] > 0.5)
            
            # Short: price < KAMA, RSI < 50, chop < 61.8 (trending), weekly downtrend
            short_cond = (close[i] < kama_values[i] and 
                          rsi_values[i] < 50 and 
                          chop_values[i] < 61.8 and 
                          trend_1w_aligned[i] < 0.5)
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price < KAMA or RSI < 40
            if close[i] < kama_values[i] or rsi_values[i] < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price > KAMA or RSI > 60
            if close[i] > kama_values[i] or rsi_values[i] > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: KAMA trend + RSI momentum + chop filter on daily timeframe with weekly trend filter.
# KAMA adapts to market noise, reducing false signals in choppy markets.
# RSI confirms momentum direction while chop filter ensures we only trade in trending markets (chop < 61.8).
# Weekly EMA34 ensures alignment with higher timeframe trend.
# Works in bull markets (trend following) and bear markets (avoids false signals in chop, captures trends).
# Discrete sizing (0.25) minimizes churn. Targets 15-25 trades/year on daily timeframe.