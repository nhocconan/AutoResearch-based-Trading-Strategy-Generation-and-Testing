#!/usr/bin/env python3
"""
4h_cci_trend_volume_v1
Hypothesis: CCI(20) identifies overbought/oversold extremes (>100 or <-100) for mean reversion trades.
In ranging markets (ADX < 25), fade extremes with volume confirmation. In trending markets (ADX >= 25),
avoid counter-trend trades to reduce whipsaw. Uses 1d EMA50 for trend filter and volume spike confirmation.
Designed to work in both bull and bear markets by adapting to volatility regime via ADX filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_cci_trend_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    ema50_4h = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # CCI(20) calculation
    typical_price = (high + low + close) / 3
    sma_tp = pd.Series(typical_price).rolling(window=20, min_periods=20).mean().values
    mad_tp = pd.Series(typical_price).rolling(window=20, min_periods=20).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=True).values
    # Avoid division by zero
    cci = np.where(mad_tp != 0, (typical_price - sma_tp) / (0.015 * mad_tp), 0)
    
    # ADX(14) for trend strength
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            high_diff = high[i] - high[i-1]
            low_diff = low[i-1] - low[i]
            plus_dm[i] = high_diff if high_diff > low_diff and high_diff > 0 else 0
            minus_dm[i] = low_diff if low_diff > high_diff and low_diff > 0 else 0
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Wilder's smoothing
        atr = np.zeros_like(tr)
        atr[period] = np.mean(tr[1:period+1])
        for i in range(period+1, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        plus_di = np.zeros_like(high)
        minus_di = np.zeros_like(high)
        dx = np.zeros_like(high)
        
        for i in range(period, len(high)):
            plus_di[i] = 100 * (np.mean(plus_dm[i-period+1:i+1]) / atr[i]) if atr[i] != 0 else 0
            minus_di[i] = 100 * (np.mean(minus_dm[i-period+1:i+1]) / atr[i]) if atr[i] != 0 else 0
            dx[i] = (abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])) * 100 if (plus_di[i] + minus_di[i]) != 0 else 0
        
        adx = np.zeros_like(high)
        adx[2*period-1] = np.mean(dx[period:2*period])
        for i in range(2*period, len(dx)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        return adx
    
    adx = calculate_adx(high, low, close, 14)
    
    # 20-period volume average
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(ema50_4h[i]) or np.isnan(cci[i]) or 
            np.isnan(adx[i]) or np.isnan(vol_sma[i]) or
            np.isnan(close[i]) or np.isnan(high[i]) or np.isnan(low[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x average volume
        vol_confirm = volume[i] > 1.3 * vol_sma[i]
        
        if position == 1:  # Long position
            # Exit: CCI returns below 0 (mean reversion complete) OR 
            # CCI drops below -100 in strong trend (trend continuation)
            if cci[i] < 0 or (adx[i] >= 25 and cci[i] < -100):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: CCI returns above 0 (mean reversion complete) OR
            # CCI rises above 100 in strong trend (trend continuation)
            if cci[i] > 0 or (adx[i] >= 25 and cci[i] > 100):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Only trade in ranging markets (ADX < 25) to avoid trend-following whipsaw
            if adx[i] < 25:
                # Mean reversion long at CCI < -100 (oversold) with volume
                if (cci[i] <= -100 and 
                    vol_confirm and 
                    close[i] > ema50_4h[i]):  # Above daily EMA for quality
                    position = 1
                    signals[i] = 0.25
                # Mean reversion short at CCI > 100 (overbought) with volume
                elif (cci[i] >= 100 and 
                      vol_confirm and 
                      close[i] < ema50_4h[i]):  # Below daily EMA for quality
                    position = -1
                    signals[i] = -0.25
    
    return signals