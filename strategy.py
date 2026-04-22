# 1D_KAMA_Trend_RSI_Chop_Filter
# Hypothesis: Daily KAMA (Kaufman Adaptive Moving Average) identifies trend direction and adapts to volatility.
# Long when price > KAMA, RSI > 50, and choppy market (CHOP > 61.8) for mean reversion to upside.
# Short when price < KAMA, RSI < 50, and choppy market (CHOP > 61.8) for mean reversion to downside.
# Uses 1-week trend filter (EMA50) to avoid counter-trend trades in strong trends.
# Designed for low trade frequency (~10-25/year) by requiring multiple confirmations and chop regime filter.
# Works in both bull and bear markets by trading mean reversion within choppy regimes while respecting higher timeframe trend.

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
    volume = prices['volume'].values
    
    # KAMA (Kaufman Adaptive Moving Average)
    def kama(close, er_length=10, fast_sc=2, slow_sc=30):
        # Efficiency Ratio
        change = np.abs(np.diff(close, prepend=close[0]))
        volatility = np.abs(np.diff(close)).cumsum()
        volatility = np.concatenate([[0], volatility[1:]])
        er = np.zeros_like(close)
        for i in range(len(close)):
            if volatility[i] > 0:
                er[i] = change[i] / volatility[i]
            else:
                er[i] = 0
        
        # Smoothing Constants
        sc = (er * (2/(fast_sc+1) - 2/(slow_sc+1)) + 2/(slow_sc+1)) ** 2
        
        # KAMA calculation
        kama_vals = np.zeros_like(close)
        kama_vals[0] = close[0]
        for i in range(1, len(close)):
            kama_vals[i] = kama_vals[i-1] + sc[i] * (close[i] - kama_vals[i-1])
        return kama_vals
    
    kama_vals = kama(close, 10, 2, 30)
    
    # RSI (14-period)
    def rsi(close, period=14):
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.zeros_like(close)
        avg_loss = np.zeros_like(close)
        
        # Initial average
        avg_gain[period] = np.mean(gain[1:period+1])
        avg_loss[period] = np.mean(loss[1:period+1])
        
        # Wilder smoothing
        for i in range(period+1, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
        rsi_vals = 100 - (100 / (1 + rs))
        return rsi_vals
    
    rsi_vals = rsi(close, 14)
    
    # Choppiness Index (14-period)
    def choppiness_index(high, low, close, period=14):
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = high[0] - low[0]  # First TR
        
        # Sum of True Ranges
        atr_sum = np.zeros_like(close)
        for i in range(len(close)):
            if i < period:
                atr_sum[i] = np.sum(tr[:i+1])
            else:
                atr_sum[i] = np.sum(tr[i-period+1:i+1])
        
        # Highest High and Lowest Low over period
        highest_high = np.zeros_like(close)
        lowest_low = np.zeros_like(close)
        for i in range(len(close)):
            if i < period:
                highest_high[i] = np.max(high[:i+1])
                lowest_low[i] = np.min(low[:i+1])
            else:
                highest_high[i] = np.max(high[i-period+1:i+1])
                lowest_low[i] = np.min(low[i-period+1:i+1])
        
        # Choppiness Formula
        chi = np.zeros_like(close)
        for i in range(len(close)):
            if atr_sum[i] > 0 and highest_high[i] != lowest_low[i]:
                chi[i] = 100 * np.log10(atr_sum[i] / (highest_high[i] - lowest_low[i])) / np.log10(period)
            else:
                chi[i] = 50  # Neutral when undefined
        return chi
    
    chop_vals = choppiness_index(high, low, close, 14)
    
    # Load weekly data for trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 50-period EMA on weekly close for trend
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(kama_vals[i]) or np.isnan(rsi_vals[i]) or np.isnan(chop_vals[i]) or 
            np.isnan(ema50_1w_aligned[i])):
            continue
        
        # Chop regime filter: only trade in choppy markets (CHOP > 61.8)
        if chop_vals[i] <= 61.8:
            # Exit position if market becomes trending
            signals[i] = 0.0
            continue
        
        # Long: price > KAMA, RSI > 50, and weekly EMA50 rising
        if (close[i] > kama_vals[i] and 
            rsi_vals[i] > 50 and 
            ema50_1w_aligned[i] > ema50_1w_aligned[i-1]):
            signals[i] = 0.25
        # Short: price < KAMA, RSI < 50, and weekly EMA50 falling
        elif (close[i] < kama_vals[i] and 
              rsi_vals[i] < 50 and 
              ema50_1w_aligned[i] < ema50_1w_aligned[i-1]):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "1D_KAMA_Trend_RSI_Chop_Filter"
timeframe = "1d"
leverage = 1.0