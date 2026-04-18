#!/usr/bin/env python3
"""
4h_KAMA_Trend_With_RSI_Filter_v2
Hypothesis: KAMA adapts to market noise, providing reliable trend direction in both trending and ranging markets. Combined with RSI(14) for momentum confirmation and volume spike filter, this strategy aims to capture medium-term trend moves with reduced whipsaw. Uses 12h EMA34 as higher timeframe trend filter for alignment. Designed for low trade frequency (~25-40/year) to minimize fee drag while maintaining edge in bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # KAMA ( Kaufman Adaptive Moving Average )
    close_series = pd.Series(close)
    # Efficiency Ratio
    change = abs(close_series - close_series.shift(10))
    volatility = abs(close_series.diff()).rolling(window=10, min_periods=10).sum()
    er = change / volatility.replace(0, np.nan)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = close_series.copy()
    kama.iloc[0] = close_series.iloc[0]
    for i in range(1, len(close_series)):
        kama.iloc[i] = kama.iloc[i-1] + sc.iloc[i] * (close_series.iloc[i] - kama.iloc[i-1])
    kama_values = kama.values
    
    # RSI(14)
    delta = close_series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Volume spike: >1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma)
    
    # 12h EMA34 trend filter (higher timeframe trend)
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 40  # Need KAMA, RSI, volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(kama_values[i]) or 
            np.isnan(rsi_values[i]) or
            np.isnan(volume_spike[i]) or
            np.isnan(ema_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        kama_val = kama_values[i]
        rsi_val = rsi_values[i]
        vol_spike = volume_spike[i]
        ema_12h_val = ema_12h_aligned[i]
        
        if position == 0:
            # Long: price > KAMA, RSI > 50, volume spike, above 12h EMA34
            if price > kama_val and rsi_val > 50 and vol_spike and price > ema_12h_val:
                signals[i] = 0.25
                position = 1
            # Short: price < KAMA, RSI < 50, volume spike, below 12h EMA34
            elif price < kama_val and rsi_val < 50 and vol_spike and price < ema_12h_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price < KAMA or RSI < 40
            if price < kama_val or rsi_val < 40:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price > KAMA or RSI > 60
            if price > kama_val or rsi_val > 60:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_KAMA_Trend_With_RSI_Filter_v2"
timeframe = "4h"
leverage = 1.0