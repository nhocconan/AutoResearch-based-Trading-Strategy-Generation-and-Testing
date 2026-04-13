#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Elder Ray + 1d ADX regime filter
    # Long when: Bear Power < 0 (bulls in control) AND ADX > 25 (trending) AND Close > EMA(20)
    # Short when: Bull Power > 0 (bears in control) AND ADX > 25 (trending) AND Close < EMA(20)
    # Exit when: Power values reverse OR ADX < 20 (range market)
    # Uses Elder Ray to measure bull/bear power via EMA(13), ADX for regime filtering.
    # Works in bull/bear via ADX regime filter avoiding whipsaws in ranging markets.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate EMA(13) for Elder Ray
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema13  # Bulls' ability to push price above EMA
    bear_power = low - ema13   # Bears' ability to push price below EMA
    
    # Calculate ADX(14) for regime filter
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            plus_dm[i] = max(0, high[i] - high[i-1])
            minus_dm[i] = max(0, low[i-1] - low[i])
            if plus_dm[i] == minus_dm[i]:
                plus_dm[i] = 0
                minus_dm[i] = 0
            elif plus_dm[i] < minus_dm[i]:
                plus_dm[i] = 0
            else:
                minus_dm[i] = 0
            
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Smooth using Wilder's smoothing (alpha = 1/period)
        def WilderSmooth(data, period):
            result = np.zeros_like(data)
            result[period-1] = np.nansum(data[:period])
            for i in range(period, len(data)):
                result[i] = result[i-1] - (result[i-1] / period) + data[i]
            return result
        
        if len(high) < period:
            return np.full_like(high, np.nan)
        
        atr = WilderSmooth(tr, period)
        plus_di = 100 * WilderSmooth(plus_dm, period) / atr
        minus_di = 100 * WilderSmooth(minus_dm, period) / atr
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = WilderSmooth(dx, period)
        return adx
    
    adx = calculate_adx(high, low, close, 14)
    
    # Get 1d data for additional regime filter (optional: weekly trend via EMA(50))
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(adx[i]) or np.isnan(ema50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime filters
        trending_market = adx[i] > 25
        ranging_market = adx[i] < 20
        bullish_bias = close[i] > ema50_1d_aligned[i]
        bearish_bias = close[i] < ema50_1d_aligned[i]
        
        # Entry conditions
        long_entry = (bear_power[i] < 0) and trending_market and bullish_bias and (position != 1)
        short_entry = (bull_power[i] > 0) and trending_market and bearish_bias and (position != -1)
        
        # Exit conditions: power reversal OR ranging market
        exit_long = (bear_power[i] >= 0) or ranging_market
        exit_short = (bull_power[i] <= 0) or ranging_market
        
        # Execute signals
        if long_entry:
            position = 1
            signals[i] = position_size
        elif short_entry:
            position = -1
            signals[i] = -position_size
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_1d_elder_ray_adx_regime_v1"
timeframe = "6h"
leverage = 1.0