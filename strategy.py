# 1d_KAMA_RSI_ChopFilter_Strategy
# KAMA trend direction + RSI(14) extremes + Choppiness regime filter
# KAMA adapts to market noise (trending vs ranging) - effective in both bull/bear
# RSI extremes for mean reversion in chop, trend alignment in trending markets
# Target: 15-25 trades/year with low churn for 1d timeframe

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend context
    weekly = get_htf_data(prices, '1w')
    weekly_close = weekly['close'].values
    
    # KAMA (Kaufman Adaptive Moving Average) - adapts to market noise
    # ER = Efficiency Ratio = |change| / volatility
    change = np.abs(np.diff(close, k=10))  # 10-period net change
    volatility = np.sum(np.abs(np.diff(close)), axis=1)  # 10-period volatility
    er = np.zeros_like(change)
    mask = volatility != 0
    er[mask] = change[mask] / volatility[mask]
    # Add leading zeros for alignment
    er = np.concatenate([np.full(10, np.nan), er])
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)  # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # Seed
    for i in range(10, n):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # Align weekly KAMA to daily
    weekly_kama = np.full_like(weekly_close, np.nan)
    # Calculate weekly KAMA (simplified)
    weekly_change = np.abs(np.diff(weekly_close, k=5))
    weekly_volatility = np.sum(np.abs(np.diff(weekly_close)), axis=1)
    weekly_er = np.zeros_like(weekly_change)
    w_mask = weekly_volatility != 0
    weekly_er[w_mask] = weekly_change[w_mask] / weekly_volatility[w_mask]
    weekly_er = np.concatenate([np.full(5, np.nan), weekly_er])
    weekly_fast_sc = 2 / (2 + 1)
    weekly_slow_sc = 2 / (30 + 1)
    weekly_sc = (weekly_er * (weekly_fast_sc - weekly_slow_sc) + weekly_slow_sc) ** 2
    for i in range(5, len(weekly_close)):
        if not np.isnan(weekly_sc[i]):
            weekly_kama[i] = weekly_kama[i-1] + weekly_sc[i] * (weekly_close[i] - weekly_kama[i-1])
        else:
            weekly_kama[i] = weekly_kama[i-1]
    weekly_kama_aligned = align_htf_to_ltf(prices, weekly, weekly_kama)
    
    # RSI(14)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.concatenate([np.full(14, np.nan), rsi])
    
    # Choppiness Index - detects ranging vs trending markets
    def true_range(high, low, close_prev):
        tr1 = high - low
        tr2 = np.abs(high - close_prev)
        tr3 = np.abs(low - close_prev)
        return np.maximum(tr1, np.maximum(tr2, tr3))
    
    tr = np.concatenate([[np.nan], true_range(high[1:], low[1:], close[:-1])])
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    max_h = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_l = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    chop = np.full_like(close, np.nan)
    for i in range(13, n):
        if not np.isnan(atr14[i]) and max_h[i] > min_l[i]:
            chop[i] = 100 * np.log10(np.sum(atr14[i-13:i+1]) / (max_h[i] - min_l[i])) / np.log10(14)
        else:
            chop[i] = 50.0  # Neutral when undefined
    
    # Volume filter - avoid low liquidity
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(kama[i]) or np.isnan(weekly_kama_aligned[i]) or 
            np.isnan(rsi[i]) or np.isnan(chop[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Only trade when volume filter passes
        if volume_filter[i]:
            # Determine market regime: chop > 61.8 = ranging, chop < 38.2 = trending
            if chop[i] > 61.8:  # Ranging market - mean reversion
                # Long when RSI oversold and price above KAMA (bullish bias)
                if rsi[i] < 30 and close[i] > kama[i]:
                    signals[i] = 0.25
                # Short when RSI overbought and price below KAMA (bearish bias)
                elif rsi[i] > 70 and close[i] < kama[i]:
                    signals[i] = -0.25
                else:
                    signals[i] = 0.0  # Flat in ranging
            else:  # Trending market - follow trend
                # Long when price above both daily and weekly KAMA
                if close[i] > kama[i] and close[i] > weekly_kama_aligned[i]:
                    signals[i] = 0.25
                # Short when price below both daily and weekly KAMA
                elif close[i] < kama[i] and close[i] < weekly_kama_aligned[i]:
                    signals[i] = -0.25
                else:
                    signals[i] = 0.0  # Flat/no clear trend
        else:
            signals[i] = 0.0  # No trade on low volume
    
    return signals

name = "1d_KAMA_RSI_ChopFilter_Strategy"
timeframe = "1d"
leverage = 1.0