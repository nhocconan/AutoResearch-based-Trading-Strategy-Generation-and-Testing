#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA trend + 1w volume confirmation + ATR volatility filter
# - KAMA (Kaufman Adaptive Moving Average) from 1d: adapts to market noise, effective in both trending and ranging markets
# - Long when price > KAMA with volume > 1.5x 20-period 1w average volume
# - Short when price < KAMA with volume > 1.5x 20-period 1w average volume
# - ATR filter: only trade when ATR(10) > 0.3 * ATR(30) to avoid low volatility chop
# - Uses discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 20-50 trades/year (80-200 total over 4 years) to stay within fee drag limits
# - KAMA works in bull markets by following trends and in bear markets by quickly adapting to downtrends
# - 1w HTF provides reliable volume confirmation, reducing false breakouts

name = "1d_1w_kama_volume_atr_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1w data ONCE before loop for volume confirmation and ATR
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return signals
    
    # Pre-compute 1w volume SMA and ATR
    volume_1w = df_1w['volume'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True range for ATR
    tr1 = pd.Series(high_1w).shift(1) - pd.Series(low_1w).shift(1)
    tr2 = abs(pd.Series(high_1w).shift(1) - pd.Series(close_1w).shift(1))
    tr3 = abs(pd.Series(low_1w).shift(1) - pd.Series(close_1w).shift(1))
    tr_1w = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_10_1w = pd.Series(tr_1w).ewm(span=10, adjust=False, min_periods=10).mean().values
    atr_30_1w = pd.Series(tr_1w).ewm(span=30, adjust=False, min_periods=30).mean().values
    
    # 1w volume SMA (20-period)
    volume_series = pd.Series(volume_1w)
    volume_sma_20_1w = volume_series.rolling(window=20, min_periods=20).mean().values
    
    # Align 1w indicators to 1d timeframe
    volume_sma_20_aligned = align_htf_to_ltf(prices, df_1w, volume_sma_20_1w)
    atr_10_aligned = align_htf_to_ltf(prices, df_1w, atr_10_1w)
    atr_30_aligned = align_htf_to_ltf(prices, df_1w, atr_30_1w)
    
    # Pre-compute 1d KAMA (Kaufman Adaptive Moving Average)
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close, n=10))  # |close - close[10]|
    volatility = np.sum(np.abs(np.diff(close)), axis=1)  # sum of |close - close[1]| over 10 periods
    # Avoid division by zero
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    fastest = 2 / (2 + 1)   # for EMA period 2
    slowest = 2 / (30 + 1)  # for EMA period 30
    sc = (er * (fastest - slowest) + slowest) ** 2
    # Calculate KAMA
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # seed value
    for i in range(10, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(kama[i]) or np.isnan(volume_sma_20_aligned[i]) or
            np.isnan(atr_10_aligned[i]) or np.isnan(atr_30_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        price_close = close[i]
        volume_current = volume[i]
        
        # KAMA trend conditions
        price_above_kama = price_close > kama[i]
        price_below_kama = price_close < kama[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average (using 1w aligned volume)
        vol_confirm = volume_current > 1.5 * volume_sma_20_aligned[i]
        
        # ATR filter: trade only when short-term ATR > 0.3 * long-term ATR (avoid low volatility)
        atr_filter = atr_10_aligned[i] > 0.3 * atr_30_aligned[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: price above KAMA + volume confirmation + ATR filter
        if price_above_kama and vol_confirm and atr_filter:
            enter_long = True
        
        # Short: price below KAMA + volume confirmation + ATR filter
        if price_below_kama and vol_confirm and atr_filter:
            enter_short = True
        
        # Exit conditions: opposite KAMA touch or volatility collapse
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price touches/below KAMA OR volatility collapses
            exit_long = (not price_above_kama) or (not atr_filter)
        elif position == -1:
            # Exit short if price touches/above KAMA OR volatility collapses
            exit_short = (not price_below_kama) or (not atr_filter)
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals