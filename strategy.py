#!/usr/bin/env python3
"""
12h_RSI_VWAP_Trend_Follow
Hypothesis: On 12h timeframe, follow the trend using 1d EMA34, with entries triggered by RSI(14) pullbacks in the direction of the trend.
Volume confirmation via VWAP > price for longs and VWAP < price for shorts.
Works in both bull and bear markets by only taking trend-aligned trades.
Designed for low trade frequency (target: 20-40 trades/year) to minimize fee drag.
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
    
    # === 1d data for EMA trend, RSI, and VWAP ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 1d EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # 1d RSI(14) for entry signals
    delta = pd.Series(close_1d).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d = rsi_1d.values
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # 1d VWAP (volume-weighted average price)
    # Typical price = (H+L+C)/3
    typical_price_1d = (high_1d + low_1d + close_1d) / 3.0
    vp = typical_price_1d * volume_1d
    cum_vp = np.nancumsum(vp)
    cum_vol = np.nancumsum(volume_1d)
    vwap_1d = np.divide(cum_vp, cum_vol, out=np.zeros_like(cum_vp), where=cum_vol!=0)
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    
    signals = np.zeros(n)
    
    # Warmup: covers EMA34 and RSI
    warmup = 40
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(vwap_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Trend filter: price above/below EMA34
        bullish_trend = close[i] > ema34_1d_aligned[i]
        bearish_trend = close[i] < ema34_1d_aligned[i]
        
        # VWAP filter: price above VWAP for longs, below for shorts
        price_above_vwap = close[i] > vwap_1d_aligned[i]
        price_below_vwap = close[i] < vwap_1d_aligned[i]
        
        # RSI conditions for pullback entries
        rsi_oversold = rsi_1d_aligned[i] < 30
        rsi_overbought = rsi_1d_aligned[i] > 70
        
        if position == 0:
            # Long: bullish trend + price above VWAP + RSI oversold
            if bullish_trend and price_above_vwap and rsi_oversold:
                signals[i] = 0.25
                position = 1
                continue
            
            # Short: bearish trend + price below VWAP + RSI overbought
            if bearish_trend and price_below_vwap and rsi_overbought:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit conditions: trend reversal or RSI returns to neutral
        elif position == 1:
            # Exit long when trend turns bearish or RSI reaches overbought
            if not bullish_trend or rsi_1d_aligned[i] > 70:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short when trend turns bullish or RSI reaches oversold
            if not bearish_trend or rsi_1d_aligned[i] < 30:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_RSI_VWAP_Trend_Follow"
timeframe = "12h"
leverage = 1.0