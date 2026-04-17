#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA trend with 1w EMA filter and volume confirmation.
# Uses Kaufman's Adaptive Moving Average (KAMA) for trend direction,
# 1-week EMA for higher timeframe trend filter, and volume spike for confirmation.
# Designed to work in bull (KAMA up, price above 1w EMA) and bear (KAMA down, price below 1w EMA).
# Target: 10-25 trades/year to avoid fee drag on daily timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate KAMA (10-period ER, 2/30 for fast/slow SC)
    close_series = pd.Series(close)
    change = abs(close_series - close_series.shift(10))
    volatility = abs(close_series.diff()).rolling(window=10, min_periods=1).sum()
    er = change / volatility.replace(0, np.nan)
    sc = (er * (2/2 - 2/30) + 2/30) ** 2
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        if not np.isnan(sc.iloc[i]):
            kama[i] = kama[i-1] + sc.iloc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # Calculate 1w EMA50 for trend filter
    close_1w_series = pd.Series(close_1w)
    ema50_1w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1w EMA to 1d
    ema50_1d = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume filter: current volume > 1.5 * 20-day average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 30  # Need KAMA (10) + EMA50 (1w) + volume MA20
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(kama[i]) or 
            np.isnan(ema50_1d[i]) or 
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: spike > 1.5x average
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        
        # KAMA trend direction
        kama_rising = kama[i] > kama[i-1]
        kama_falling = kama[i] < kama[i-1]
        
        # Price relative to 1w EMA50
        price_above_ema = close[i] > ema50_1d[i]
        price_below_ema = close[i] < ema50_1d[i]
        
        if position == 0:
            # Long: KAMA rising AND price above 1w EMA50 AND volume confirmation
            if (kama_rising and price_above_ema and volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: KAMA falling AND price below 1w EMA50 AND volume confirmation
            elif (kama_falling and price_below_ema and volume_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: KAMA falls OR price crosses below 1w EMA50
            if (not kama_rising) or (close[i] < ema50_1d[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: KAMA rises OR price crosses above 1w EMA50
            if (not kama_falling) or (close[i] > ema50_1d[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_KAMA_1wEMA50_Volume"
timeframe = "1d"
leverage = 1.0