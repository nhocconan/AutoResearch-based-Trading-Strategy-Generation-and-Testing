#!/usr/bin/env python3
"""
1d_KAMA_Trend_With_RSI_Filter_and_Chop_Regime
Hypothesis: On daily timeframe, use Kaufman Adaptive Moving Average (KAMA) for trend direction,
combined with RSI(14) for momentum confirmation and Choppiness Index(14) to avoid ranging markets.
Only trade when KAMA slope aligns with RSI extremes and market is trending (CHOP < 38.2 or > 61.8).
This combines trend-following with momentum and regime filtering to work in both bull and bear markets.
Target: 15-25 trades/year per symbol to minimize fee drag while capturing major moves.
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
    
    # Get weekly data for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate weekly EMA34 for HTF trend
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate KAMA on daily close
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close, n=10))  # 10-period net change
    volatility = np.sum(np.abs(np.diff(close)), axis=1)  # 10-period sum of absolute changes
    # Pad the beginning with NaN for alignment
    change = np.concatenate([np.full(9, np.nan), change])
    volatility = np.concatenate([np.full(9, np.nan), volatility])
    # Avoid division by zero
    er = np.where(volatility > 0, change / volatility, 0)
    # Smoothing constants
    fast_sc = 2 / (2 + 1)  # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    # Initialize KAMA
    kama = np.full(n, np.nan)
    kama[9] = close[9]  # Start after 10 periods
    for i in range(10, n):
        if np.isnan(kama[i-1]) or np.isnan(sc[i]):
            kama[i] = close[i]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate RSI(14)
    delta = np.diff(close)
    delta = np.concatenate([[np.nan], delta])  # align with close
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    # Wilder's smoothing
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    avg_gain[14] = np.nanmean(gain[1:15])  # first average
    avg_loss[14] = np.nanmean(loss[1:15])
    for i in range(15, n):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate Choppiness Index(14)
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.concatenate([[np.nan], close[:-1]]))
    tr3 = np.abs(low - np.concatenate([[np.nan], close[:-1]]))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    # Sum of ATR over 14 periods
    atr_sum = np.full(n, np.nan)
    for i in range(13, n):
        atr_sum[i] = np.sum(tr[i-13:i+1])
    # Highest high and lowest low over 14 periods
    max_high = np.full(n, np.nan)
    min_low = np.full(n, np.nan)
    for i in range(13, n):
        max_high[i] = np.max(high[i-13:i+1])
        min_low[i] = np.min(low[i-13:i+1])
    # Chop = 100 * log10(sum(tr) / (max_high - min_low)) / log10(14)
    range_hl = max_high - min_low
    chop = np.full(n, np.nan)
    for i in range(13, n):
        if range_hl[i] > 0:
            chop[i] = 100 * np.log10(atr_sum[i] / range_hl[i]) / np.log10(14)
        else:
            chop[i] = 50  # middle value when no range
    
    # Align HTF EMA34
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need KAMA(10), RSI(14), CHOP(14), EMA34(1w)
    start_idx = max(34, 14, 13)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama[i]) or 
            np.isnan(rsi[i]) or 
            np.isnan(chop[i]) or
            np.isnan(ema_34_1w_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Trend: price relative to KAMA
        price_above_kama = close[i] > kama[i]
        price_below_kama = close[i] < kama[i]
        # KAMA slope (direction)
        kama_rising = kama[i] > kama[i-1]
        kama_falling = kama[i] < kama[i-1]
        
        # Regime: trending market (CHOP < 38.2 or CHOP > 61.8)
        trending_market = (chop[i] < 38.2) or (chop[i] > 61.8)
        
        # Momentum: RSI extremes
        rsi_oversold = rsi[i] < 30
        rsi_overbought = rsi[i] > 70
        
        if position == 0:
            # Long: KAMA rising + price above KAMA + RSI oversold + trending market
            long_signal = kama_rising and price_above_kama and rsi_oversold and trending_market
            # Short: KAMA falling + price below KAMA + RSI overbought + trending market
            short_signal = kama_falling and price_below_kama and rsi_overbought and trending_market
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: KAMA falling OR RSI overbought OR market becomes ranging
            if (not kama_rising) or rsi_overbought or (chop[i] >= 38.2 and chop[i] <= 61.8):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: KAMA rising OR RSI oversold OR market becomes ranging
            if (not kama_falling) or rsi_oversold or (chop[i] >= 38.2 and chop[i] <= 61.8):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_KAMA_Trend_With_RSI_Filter_and_Chop_Regime"
timeframe = "1d"
leverage = 1.0