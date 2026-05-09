#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d daily close outside Bollinger Bands (20,2) with volume confirmation (>1.5x 20 EMA volume) and 1w EMA34 trend filter.
# Bollinger breakouts capture volatility expansion; volume confirms institutional interest; 1w EMA34 ensures alignment with higher timeframe trend.
# Works in bull markets (breakouts above upper band) and bear markets (breakdowns below lower band).
# Uses Bollinger Bands for volatility-based breakout detection, which adapts to changing market conditions.
name = "1d_BollingerBreakout_1wEMA34_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Bollinger Bands (20,2)
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_band = sma_20 + (2 * std_20)
    lower_band = sma_20 - (2 * std_20)
    
    # 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # 1w EMA34 trend filter
    ema_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Volume confirmation: volume > 1.5x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_ema20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if (np.isnan(sma_20[i]) or np.isnan(std_20[i]) or np.isnan(ema_1w_aligned[i]) or 
            np.isnan(vol_ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: close > upper Bollinger Band + volume confirmation + 1w EMA34 up
            if (price > upper_band[i] and vol_confirm[i] and price > ema_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: close < lower Bollinger Band + volume confirmation + 1w EMA34 down
            elif (price < lower_band[i] and vol_confirm[i] and price < ema_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: close crosses below middle Bollinger Band (SMA20)
            if price < sma_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: close crosses above middle Bollinger Band (SMA20)
            if price > sma_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals