#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA trend + RSI mean reversion + chop regime filter for 1d timeframe.
# Long: KAMA rising (uptrend) AND RSI(14) < 30 (oversold) AND CHOP(14) > 61.8 (rangy market)
# Short: KAMA falling (downtrend) AND RSI(14) > 70 (overbought) AND CHOP(14) > 61.8 (rangy market)
# Exit: Opposite RSI condition or KAMA trend reversal.
# Uses 1w HTF for EMA34 trend filter to avoid counter-trend trades in strong weekly trends.
# Discrete sizing 0.25. Target: 40-80 total trades over 4 years (10-20/year).
# KAMA adapts to market noise, RSI captures mean reversion in chop, weekly EMA filters major trend.
# Works in bull via long signals in uptrend + oversold, and in bear via short signals in downtrend + overbought.

name = "1d_KAMA_RSI_Chop_1wEMA34"
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
    
    # Calculate KAMA (adaptive moving average) for trend
    close_s = pd.Series(close)
    # Efficiency Ratio: |net change| / sum of absolute changes over 10 periods
    change = abs(close - close.shift(10))
    volatility = abs(close - close.shift(1)).rolling(window=10, min_periods=10).sum()
    er = change / volatility.replace(0, np.nan)
    # Smoothing constants: fastest EMA(2), slowest EMA(30)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # Fill NaN with slowest SC (when volatility=0)
    sc = sc.fillna(2/(30+1)) ** 2
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        if np.isnan(sc.iloc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc.iloc[i] * (close[i] - kama[i-1])
    
    # Calculate RSI(14)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values  # neutral when undefined
    
    # Calculate Choppiness Index(14)
    atr = pd.Series(np.maximum(high - low, np.maximum(abs(high - close_s.shift()), abs(low - close_s.shift()))))
    atr_sum = atr.rolling(window=14, min_periods=14).sum()
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min()
    chop = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(14)
    chop = chop.replace([np.inf, -np.inf], np.nan).fillna(50).values  # neutral when undefined
    
    # Get 1w data for EMA34 trend filter (higher timeframe)
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 1w EMA34 for trend filter
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or 
            np.isnan(ema_34_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        kama_val = kama[i]
        rsi_val = rsi[i]
        chop_val = chop[i]
        ema_trend = ema_34_1w_aligned[i]
        
        # Determine KAMA trend (rising/falling)
        kama_rising = kama_val > kama[i-1] if i > 0 else False
        kama_falling = kama_val < kama[i-1] if i > 0 else False
        
        # Determine regime: choppy market (good for mean reversion)
        is_choppy = chop_val > 61.8
        
        # Entry logic
        if position == 0:
            # Long: KAMA rising (uptrend) AND RSI < 30 (oversold) AND choppy market
            if kama_rising and rsi_val < 30 and is_choppy:
                signals[i] = 0.25
                position = 1
            # Short: KAMA falling (downtrend) AND RSI > 70 (overbought) AND choppy market
            elif kama_falling and rsi_val > 70 and is_choppy:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: RSI > 50 (mean reversion complete) OR KAMA turns down
            if rsi_val > 50 or not kama_rising:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: RSI < 50 (mean reversion complete) OR KAMA turns up
            if rsi_val < 50 or not kama_falling:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals