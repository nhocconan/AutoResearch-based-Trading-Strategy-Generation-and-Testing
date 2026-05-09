#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h momentum strategy using 1-day RSI divergence and price action at key levels.
# In bull markets: buy on bullish RSI divergence near support; in bear markets: sell on bearish RSI divergence near resistance.
# Uses 1-day RSI for momentum divergence detection and price position relative to 1-day VWAP for entry timing.
# Designed to capture swing points with low frequency to avoid fee drag.
name = "12h_RSIDivergence_VWAP_Pullback"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1-day data for RSI and VWAP
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:  # Need at least 14 days for RSI
        return np.zeros(n)
    
    # Calculate 1-day RSI(14)
    delta = df_1d['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Calculate 1-day VWAP
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3.0
    vwap_numerator = (typical_price * df_1d['volume']).cumsum()
    vwap_denominator = df_1d['volume'].cumsum()
    vwap = (vwap_numerator / vwap_denominator).values
    
    # Align 1-day indicators to 12h timeframe (use previous day's values)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi_values)
    vwap_aligned = align_htf_to_ltf(prices, df_1d, vwap)
    
    # Calculate 12-period RSI on 12h for entry timing confirmation
    delta_12h = pd.Series(close).diff()
    gain_12h = delta_12h.clip(lower=0)
    loss_12h = -delta_12h.clip(upper=0)
    avg_gain_12h = gain_12h.ewm(alpha=1/12, adjust=False, min_periods=12).mean()
    avg_loss_12h = loss_12h.ewm(alpha=1/12, adjust=False, min_periods=12).mean()
    rs_12h = avg_gain_12h / avg_loss_12h
    rsi_12h = 100 - (100 / (1 + rs_12h))
    rsi_12h_values = rsi_12h.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 1  # Need at least 1 day of data
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(rsi_aligned[i]) or np.isnan(vwap_aligned[i]) or 
            np.isnan(rsi_12h_values[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        rsi_1d = rsi_aligned[i]
        rsi_12h = rsi_12h_values[i]
        vwap_1d = vwap_aligned[i]
        
        # Calculate RSI slope for divergence detection (3-period change)
        if i >= 3:
            rsi_slope_1d = rsi_1d - rsi_aligned[i-3]
            price_slope = price - close[i-3]
            
            # Bullish divergence: price making lower low, RSI making higher low
            bullish_div = (price_slope < 0) and (rsi_slope_1d > 0) and (rsi_1d < 40)
            # Bearish divergence: price making higher high, RSI making lower high
            bearish_div = (price_slope > 0) and (rsi_slope_1d < 0) and (rsi_1d > 60)
        else:
            bullish_div = False
            bearish_div = False
        
        if position == 0:
            # Enter long: bullish RSI divergence + price near VWAP support + 12h RSI not overbought
            if bullish_div and price >= vwap_1d * 0.995 and rsi_12h < 70:
                signals[i] = 0.25
                position = 1
            # Enter short: bearish RSI divergence + price near VWAP resistance + 12h RSI not oversold
            elif bearish_div and price <= vwap_1d * 1.005 and rsi_12h > 30:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: bearish divergence or price significantly above VWAP
            if bearish_div or price > vwap_1d * 1.02:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: bullish divergence or price significantly below VWAP
            if bullish_div or price < vwap_1d * 0.98:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals