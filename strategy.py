# #!/usr/bin/env python3
"""
6h_Liquidity_Pullback_Momentum
Hypothesis: In strong trends, price pulls back to liquidity zones (previous day's VWAP or volume-weighted areas) before continuing. Enter long when price pulls back to prior day's VWAP with bullish momentum (close > open) and volume confirmation; short when price rallies to prior day's VWAP with bearish momentum (close < open) and volume confirmation. Trend filter: price must be above/below 20-period EMA on 6h to ensure we're trading with the trend. This captures institutional order flow behavior where large players accumulate/distribute at known liquidity levels. Works in both bull (buy pullbacks) and bear (sell rallies) markets. Target: 15-25 trades/year on 6h timeframe.
"""

name = "6h_Liquidity_Pullback_Momentum"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_price = prices['open'].values
    
    # Get daily data for VWAP calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily VWAP
    typical_price_1d = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    vwap_1d = (typical_price_1d * df_1d['volume']).cumsum() / df_1d['volume'].cumsum()
    vwap_1d = vwap_1d.values
    
    # Align daily VWAP to 6h timeframe (no extra delay needed for VWAP)
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    
    # Calculate 6h EMA20 for trend filter
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(vwap_1d_aligned[i]) or np.isnan(ema_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.2x 20-period average
        vol_ma_20 = np.mean(volume[max(0, i-19):i+1]) if i >= 19 else np.mean(volume[:i+1])
        vol_confirm = volume[i] > 1.2 * vol_ma_20 if i >= 19 else volume[i] > 0
        
        # Price action: bullish/bearish candle
        bullish_candle = close[i] > open_price[i]
        bearish_candle = close[i] < open_price[i]
        
        # Distance to VWAP (normalized by ATR-like measure)
        price_to_vwap = abs(close[i] - vwap_1d_aligned[i]) / close[i]
        near_vwap = price_to_vwap < 0.005  # Within 0.5% of VWAP
        
        if position == 0:
            # LONG: Pullback to VWAP with bullish candle + volume confirmation + above EMA20
            if (near_vwap and bullish_candle and vol_confirm and 
                close[i] > ema_20[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Rally to VWAP with bearish candle + volume confirmation + below EMA20
            elif (near_vwap and bearish_candle and vol_confirm and 
                  close[i] < ema_20[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price moves below VWAP or below EMA20
            if close[i] < vwap_1d_aligned[i] or close[i] < ema_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price moves above VWAP or above EMA20
            if close[i] > vwap_1d_aligned[i] or close[i] > ema_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals