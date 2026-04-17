#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Bollinger Band width contraction/expansion with 1d ADX trend filter and volume confirmation.
# Enters long when BB width expands from low volatility AND price > BB middle AND ADX > 25 (trend up).
# Enters short when BB width expands from low volatility AND price < BB middle AND ADX > 25 (trend down).
# BB width contraction signals potential breakout; expansion confirms momentum.
# ADX filter ensures we only trade in trending markets, avoiding whipsaws in ranges.
# Volume confirmation adds conviction. Designed for low turnover (target: 20-50 trades/year).
# Works in bull markets (trend continuation) and bear markets (trend continuation down).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ADX calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 4h Bollinger Bands (20, 2)
    close_series = pd.Series(close)
    bb_middle = close_series.rolling(window=20, min_periods=20).mean().values
    bb_std = close_series.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    bb_width = bb_upper - bb_lower
    
    # Calculate BB width percentile rank (lookback 50) to detect low volatility
    # We'll use: current width < 20th percentile of past 50 = low volatility
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=50, min_periods=10).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else np.nan, raw=False
    ).values
    
    # Calculate 1d ADX (14)
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            plus_dm[i] = max(0, high[i] - high[i-1])
            minus_dm[i] = max(0, low[i-1] - low[i])
            if plus_dm[i] < minus_dm[i]:
                plus_dm[i] = 0
            if minus_dm[i] < plus_dm[i]:
                minus_dm[i] = 0
            
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i-1]),
                abs(low[i] - close[i-1])
            )
        
        # Wilder's smoothing
        atr = np.zeros_like(high)
        atr[period] = np.mean(tr[1:period+1])
        for i in range(period+1, len(high)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        plus_di = np.zeros_like(high)
        minus_di = np.zeros_like(high)
        dx = np.zeros_like(high)
        
        # Smoothed +/- DM
        plus_dm_sm = np.zeros_like(high)
        minus_dm_sm = np.zeros_like(high)
        plus_dm_sm[period] = np.sum(plus_dm[1:period+1])
        minus_dm_sm[period] = np.sum(minus_dm[1:period+1])
        
        for i in range(period+1, len(high)):
            plus_dm_sm[i] = plus_dm_sm[i-1] - (plus_dm_sm[i-1] / period) + plus_dm[i]
            minus_dm_sm[i] = minus_dm_sm[i-1] - (minus_dm_sm[i-1] / period) + minus_dm[i]
        
        # Avoid division by zero
        plus_di = np.where(atr != 0, 100 * plus_dm_sm / atr, 0)
        minus_di = np.where(atr != 0, 100 * minus_dm_sm / atr, 0)
        
        dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
        
        # ADX: smoothed DX
        adx = np.zeros_like(high)
        adx[2*period-1] = np.mean(dx[period:2*period])
        for i in range(2*period, len(high)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_1d_4h = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Volume filter: current volume > 1.5 * 20-period average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # Need sufficient data for BB width percentile and ADX
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(bb_width_percentile[i]) or 
            np.isnan(adx_1d_4h[i]) or 
            np.isnan(bb_middle[i]) or 
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: BB width in low volatility regime (< 20th percentile)
        low_volatility = bb_width_percentile[i] < 0.2
        
        # Volatility expansion: current width > previous width
        vol_expansion = bb_width[i] > bb_width[i-1] if i > 0 else False
        
        # Price relative to BB middle
        price_above_middle = close[i] > bb_middle[i]
        price_below_middle = close[i] < bb_middle[i]
        
        # Trend filter: ADX > 25 (strong trend)
        strong_trend = adx_1d_4h[i] > 25
        
        # Volume filter
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        
        if position == 0:
            # Long: Low volatility + BB width expansion + price above middle + strong uptrend + volume
            if (low_volatility and vol_expansion and price_above_middle and strong_trend and volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: Low volatility + BB width expansion + price below middle + strong downtrend + volume
            elif (low_volatility and vol_expansion and price_below_middle and strong_trend and volume_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Volatility contraction OR ADX weakens OR price crosses below BB middle
            vol_contraction = bb_width[i] < bb_width[i-1] if i > 0 else False
            weak_trend = adx_1d_4h[i] < 20
            if vol_contraction or weak_trend or (close[i] < bb_middle[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Volatility contraction OR ADX weakens OR price crosses above BB middle
            vol_contraction = bb_width[i] < bb_width[i-1] if i > 0 else False
            weak_trend = adx_1d_4h[i] < 20
            if vol_contraction or weak_trend or (close[i] > bb_middle[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_BBWidth_Expansion_ADX14_Volume"
timeframe = "4h"
leverage = 1.0