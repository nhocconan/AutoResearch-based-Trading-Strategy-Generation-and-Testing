#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Volume-Weighted MACD with 1w Trend Filter and ATR Regime
# Long when MACD line crosses above signal line with volume > 1.5x 24-bar average AND price > 1w EMA50 (uptrend) AND ATR(14) < ATR(50) (low volatility regime)
# Short when MACD line crosses below signal line with volume > 1.5x 24-bar average AND price < 1w EMA50 (downtrend) AND ATR(14) < ATR(50) (low volatility regime)
# Exit when MACD histogram crosses zero (mean reversion to momentum equilibrium)
# Target: 50-150 total trades over 4 years = 12-37/year. Uses discrete sizing (0.25) to minimize fee churn.
# Works in both bull and bear markets by combining momentum (MACD), trend (1w EMA50), and volatility regime (ATR ratio) filters.

name = "6h_VolumeWeighted_MACD_1wEMA50_ATR_Regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate ATR(14) and ATR(50) for volatility regime filter
    tr1 = np.maximum(high - low, np.abs(high - np.roll(close, 1)))
    tr2 = np.maximum(tr1, np.abs(low - np.roll(close, 1)))
    tr1[0] = high[0] - low[0]  # First bar TR
    tr2[0] = high[0] - low[0]
    atr_14 = pd.Series(tr2).rolling(window=14, min_periods=14).mean().values
    atr_50 = pd.Series(tr2).rolling(window=50, min_periods=50).mean().values
    
    # Volume confirmation (1.5x 24-period average)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().shift(1).values
    volume_spike = volume > (vol_ma * 1.5)
    
    # Calculate MACD (12,26,9)
    ema_12 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema_26 = pd.Series(close).ewm(span=26, adjust=False, min_periods=26).mean().values
    macd_line = ema_12 - ema_26
    signal_line = pd.Series(macd_line).ewm(span=9, adjust=False, min_periods=9).mean().values
    macd_histogram = macd_line - signal_line
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for all calculations)
    start_idx = max(12, 26, 9, 50, 14, 50, 24) + 1  # MACD + 1w EMA50 + ATRs + volume MA + shift(1)
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_12[i]) or np.isnan(ema_26[i]) or np.isnan(macd_line[i]) or 
            np.isnan(signal_line[i]) or np.isnan(macd_histogram[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(atr_14[i]) or np.isnan(atr_50[i]) or 
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Volatility regime: low volatility (ATR14 < ATR50) for cleaner signals
        low_vol_regime = atr_14[i] < atr_50[i]
        
        if position == 0:  # Flat - look for new entries
            # Long entry: MACD bullish crossover with volume spike, uptrend, and low volatility
            if (macd_line[i-1] <= signal_line[i-1] and  # Previous: MACD <= signal
                macd_line[i] > signal_line[i] and       # Current: MACD > signal (bullish cross)
                volume_spike[i] and 
                close[i] > ema_50_1w_aligned[i] and
                low_vol_regime):
                signals[i] = 0.25
                position = 1
            # Short entry: MACD bearish crossover with volume spike, downtrend, and low volatility
            elif (macd_line[i-1] >= signal_line[i-1] and  # Previous: MACD >= signal
                  macd_line[i] < signal_line[i] and       # Current: MACD < signal (bearish cross)
                  volume_spike[i] and 
                  close[i] < ema_50_1w_aligned[i] and
                  low_vol_regime):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: MACD histogram crosses below zero (momentum weakening)
            if macd_histogram[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: MACD histogram crosses above zero (momentum weakening)
            if macd_histogram[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals