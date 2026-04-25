#!/usr/bin/env python3
"""
1d_KAMA_Trend_RSI_Chop
Hypothesis: Daily KAMA trend direction (adaptive trend filter) combined with RSI extremes and Choppiness Index regime filter to avoid whipsaws. KAMA adapts to market noise, reducing false signals in choppy markets while capturing strong trends. RSI provides mean-reversion entries within the trend. Choppiness Index ensures we only trade when market is trending (CHOP < 38.2) or mean-reverting (CHOP > 61.8) appropriately. Designed for low trade frequency (<25/year) with strong edge in both bull and bear markets via regime adaptation.
"""

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
    
    # 1d data for indicators (primary timeframe)
    df_1d = get_htf_data(prices, '1d')
    
    # 1w data for HTF trend filter (loaded ONCE)
    df_1w = get_htf_data(prices, '1w')
    
    # KAMA (Adaptive Moving Average) - 1d
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close, n=10))
    volatility = np.sum(np.abs(np.diff(close, n=1)), axis=0)  # sum of absolute changes over 10 periods
    # Fix array lengths: change is len(close)-10, volatility is len(close)-1
    # Use rolling sum for volatility over 10 periods
    volatility_sum = pd.Series(np.abs(np.diff(close, n=1))).rolling(window=10, min_periods=10).sum().values
    # Pad change array to match length
    change_padded = np.concatenate([np.full(10, np.nan), change])
    er = np.where(volatility_sum > 0, change_padded / volatility_sum, 0)
    # Smoothing constants
    fast_sc = 2 / (2 + 1)
    slow_sc = 2 / (30 + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    # Calculate KAMA
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # seed
    for i in range(10, n):
        if not np.isnan(sc[i]) and not np.isnan(kama[i-1]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # Align KAMA to 1d (already 1d, but using align for consistency)
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # 1w EMA50 for HTF trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # RSI(14) - 1d
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    # Pad RSI to match length (first 14 values are NaN)
    rsi_padded = np.concatenate([np.full(14, np.nan), rsi])
    
    # Choppiness Index(14) - 1d
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    # Highest high and lowest low over 14 periods
    hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = np.where((hh - ll) != 0, 100 * np.log10(atr_sum / (hh - ll)) / np.log10(14), 50)
    # Pad CHOP to match length (first 14 values are NaN)
    chop_padded = np.concatenate([np.full(14, np.nan), chop])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for all indicators
    start_idx = 50  # covers KAMA seed, RSI, CHOP, EMA
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(rsi_padded[i]) or np.isnan(chop_padded[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        
        # Trend filters
        kama_trend_up = curr_close > kama_aligned[i]
        kama_trend_down = curr_close < kama_aligned[i]
        htfx_trend_up = curr_close > ema_50_1w_aligned[i]
        htfx_trend_down = curr_close < ema_50_1w_aligned[i]
        
        # Regime filters
        chop_value = chop_padded[i]
        trending_market = chop_value < 38.2  # trending regime
        ranging_market = chop_value > 61.8   # ranging regime
        
        # RSI conditions
        rsi_value = rsi_padded[i]
        rsi_oversold = rsi_value < 30
        rsi_overbought = rsi_value > 70
        rsi_neutral = 40 <= rsi_value <= 60
        
        if position == 0:
            # Look for entry signals
            # Long: KAMA uptrend + HTF uptrend + RSI oversold in ranging OR trending market
            long_entry = (kama_trend_up and htfx_trend_up and 
                         ((rsi_oversold and ranging_market) or 
                          (rsi_value < 50 and trending_market)))
            # Short: KAMA downtrend + HTF downtrend + RSI overbought in ranging OR trending market
            short_entry = (kama_trend_down and htfx_trend_down and
                          ((rsi_overbought and ranging_market) or
                           (rsi_value > 50 and trending_market)))
            
            if long_entry:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_entry:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: exit conditions
            # Stoploss: price crosses below KAMA (trend invalidation)
            # Take profit: RSI overbought OR chop indicates strong ranging (mean reversion)
            if curr_close < kama_aligned[i]:
                signals[i] = 0.0
                position = 0
            elif rsi_value > 70 or (chop_value > 61.8 and rsi_value > 50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit conditions
            # Stoploss: price crosses above KAMA (trend invalidation)
            # Take profit: RSI oversold OR chop indicates strong ranging
            if curr_close > kama_aligned[i]:
                signals[i] = 0.0
                position = 0
            elif rsi_value < 30 or (chop_value > 61.8 and rsi_value < 50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_KAMA_Trend_RSI_Chop"
timeframe = "1d"
leverage = 1.0