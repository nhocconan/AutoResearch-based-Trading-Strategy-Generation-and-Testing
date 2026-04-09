#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA trend with RSI filter and chop regime on 1w HTF
# - KAMA(10) determines trend direction on daily timeframe
# - RSI(14) > 50 for long, < 50 for short (momentum confirmation)
# - Choppiness Index(14) on 1w: < 38.2 = trending (favor signals), > 61.8 = choppy (avoid)
# - ATR(14) trailing stop: 2.5x ATR from extreme since entry
# - Position size: 0.25 (discrete level to minimize fee churn)
# - Target: 10-25 trades/year on 1d (40-100 total over 4 years)
# - Uses 1w HTF for regime filter to avoid whipsaws in ranging markets

name = "1d_kama_rsi_chop_regime_atr_v1"
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
    
    # Load 1d data ONCE before loop (primary timeframe)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Load 1w data ONCE before loop for regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # --- 1d indicators ---
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # KAMA(10) - Kaufman Adaptive Moving Average
    def calculate_kama(close, period=10, fast=2, slow=30):
        change = np.abs(np.diff(close, n=period))
        volatility = np.sum(np.abs(np.diff(close)), axis=0)
        # Handle first period values
        change = np.concatenate([np.full(period, np.nan), change])
        volatility = np.concatenate([np.full(period, np.nan), volatility])
        er = np.where(volatility != 0, change / volatility, 0)
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        kama = np.full_like(close, np.nan)
        kama[period] = close[period]
        for i in range(period+1, len(close)):
            if np.isnan(kama[i-1]):
                kama[i] = close[i]
            else:
                kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    kama = calculate_kama(close_1d, 10, 2, 30)
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # RSI(14) on 1d
    def calculate_rsi(close, period=14):
        delta = np.diff(close)
        delta = np.concatenate([[np.nan], delta])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.full_like(close, np.nan)
        avg_loss = np.full_like(close, np.nan)
        
        # First average
        avg_gain[period] = np.nanmean(gain[1:period+1])
        avg_loss[period] = np.nanmean(loss[1:period+1])
        
        # Wilder smoothing
        for i in range(period+1, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi = calculate_rsi(close_1d, 14)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # ATR(14) on 1d for stoploss
    tr1 = high_1d - low_1d
    tr2 = np.abs(np.diff(high_1d, prepend=high_1d[0]))
    tr3 = np.abs(np.diff(low_1d, prepend=low_1d[0]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # --- 1w indicators for regime filter ---
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range for 1w
    tr1_w = high_1w - low_1w
    tr2_w = np.abs(np.diff(high_1w, prepend=high_1w[0]))
    tr3_w = np.abs(np.diff(low_1w, prepend=low_1w[0]))
    tr_w = np.maximum(tr1_w, np.maximum(tr2_w, tr3_w))
    
    # Choppiness Index(14) on 1w
    def calculate_choppiness(high, low, close, period=14):
        tr = np.maximum(high - low, np.maximum(np.abs(np.diff(high, prepend=high[0])), np.abs(np.diff(low, prepend=low[0]))))
        atr = pd.Series(tr).rolling(window=period, min_periods=period).mean().values
        max_h = pd.Series(high).rolling(window=period, min_periods=period).max().values
        min_l = pd.Series(low).rolling(window=period, min_periods=period).min().values
        chop = 100 * np.log10(atr * period / (max_h - min_l)) / np.log10(period)
        return chop
    
    chop_1w = calculate_choppiness(high_1w, low_1w, close_1w, 14)
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop_1w, additional_delay_bars=0)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    highest_high_since_entry = 0.0
    lowest_low_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or
            np.isnan(atr_aligned[i]) or np.isnan(chop_aligned[i]) or
            atr_aligned[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Regime filter: only trade when not choppy (CHOP < 61.8)
        # In choppy markets (CHOP > 61.8), avoid new entries but allow exits
        if chop_aligned[i] > 61.8 and position == 0:
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Update highest high since entry
            highest_high_since_entry = max(highest_high_since_entry, high[i])
            
            # ATR-based trailing stop: exit if price drops 2.5x ATR from highest high
            if close[i] < highest_high_since_entry - 2.5 * atr_aligned[i]:
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest low since entry
            lowest_low_since_entry = min(lowest_low_since_entry, low[i])
            
            # ATR-based trailing stop: exit if price rises 2.5x ATR from lowest low
            if close[i] > lowest_low_since_entry + 2.5 * atr_aligned[i]:
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic: KAMA trend + RSI momentum + regime filter
            # Only enter when trend is aligned (price vs KAMA) AND RSI confirms momentum
            kama_bullish = close[i] > kama_aligned[i]
            kama_bearish = close[i] < kama_aligned[i]
            rsi_bullish = rsi_aligned[i] > 50
            rsi_bearish = rsi_aligned[i] < 50
            
            # Additional chop filter: avoid extreme chop
            not_extreme_chop = chop_aligned[i] < 70
            
            if not_extreme_chop:
                # Long entry: price above KAMA (bullish trend) AND RSI > 50 (bullish momentum)
                if kama_bullish and rsi_bullish:
                    position = 1
                    highest_high_since_entry = high[i]
                    lowest_low_since_entry = low[i]
                    signals[i] = 0.25
                # Short entry: price below KAMA (bearish trend) AND RSI < 50 (bearish momentum)
                elif kama_bearish and rsi_bearish:
                    position = -1
                    highest_high_since_entry = high[i]
                    lowest_low_since_entry = low[i]
                    signals[i] = -0.25
    
    return signals