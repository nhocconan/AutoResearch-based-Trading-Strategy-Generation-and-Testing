#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA Trend with Weekly Volatility Regime Filter
# Uses Kaufman Adaptive Moving Average (KAMA) to identify trend direction
# Enters long when price > KAMA and weekly volatility is low (calm market)
# Enters short when price < KAMA and weekly volatility is low
# Exits when price crosses back below/above KAMA
# Weekly volatility measured by ATR(14) percentile rank - low volatility = percentile < 40
# Designed to capture trending moves in calm markets, avoiding choppy periods
# Target: 15-30 trades per symbol over 4 years (4-7.5/year)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate KAMA (2-period ER, 30-period smoothing constant)
    # ER = |close - close[10]| / sum(|close - close[1]| over 10 periods)
    change = np.abs(np.diff(close, n=10))  # |close - close[10]|
    volatility = np.sum(np.abs(np.diff(close, n=1)), axis=1)  # sum of 1-period changes
    
    # Pad arrays for alignment
    change_padded = np.concatenate([[np.nan]*10, change])
    volatility_padded = np.concatenate([[np.nan]*10, volatility])
    
    # Calculate Efficiency Ratio
    er = np.where(volatility_padded != 0, change_padded / volatility_padded, 0)
    
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1))**2
    
    # Calculate KAMA
    kama = np.full_like(close, np.nan)
    kama[0] = close[0]
    for i in range(1, len(close)):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # Calculate weekly ATR(14) for volatility regime
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w[1:] - low_1w[1:]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # ATR(14)
    atr_1w = pd.Series(tr).ewm(span=14, adjust=False).mean().values
    
    # Percentile rank of current ATR over 50-period lookback
    atr_percentile = pd.Series(atr_1w).rolling(window=50, min_periods=10).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100 if len(x) > 0 else np.nan, raw=False
    ).values
    
    # Align indicators to daily timeframe
    kama_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), kama)
    atr_percentile_aligned = align_htf_to_ltf(prices, df_1w, atr_percentile)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 50  # for KAMA and ATR percentile calculations
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(kama_aligned[i]) or np.isnan(atr_percentile_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        if position == 0:
            # Low volatility regime: ATR percentile < 40 (calm market)
            if atr_percentile_aligned[i] < 40:
                # Long: price above KAMA
                if price > kama_aligned[i]:
                    position = 1
                    signals[i] = position_size
                # Short: price below KAMA
                elif price < kama_aligned[i]:
                    position = -1
                    signals[i] = -position_size
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0  # High volatility - stay out
        elif position == 1:
            # Exit long: price crosses below KAMA
            if price < kama_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses above KAMA
            if price > kama_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_KAMA_WeeklyVolatilityRegime"
timeframe = "1d"
leverage = 1.0