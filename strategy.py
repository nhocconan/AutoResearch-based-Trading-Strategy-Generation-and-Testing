#/usr/bin/env python3
"""
4h_price_volume_regime_v1
Hypothesis: Price channels (Donchian) breakouts in trending regimes (ADX > 25) with volume confirmation capture momentum in both bull and bear markets. 
Volatility regime filter (ATR ratio) avoids chop. Designed for 4h timeframe with tight entry conditions (~25-40 trades/year) to minimize fee drag.
Works in bull markets via breakouts and bear markets via breakdowns, filtered by higher timeframe trend and volatility regime.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_price_volume_regime_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channel (20-period)
    def calculate_donchian(high, low, period=20):
        upper = np.full(len(high), np.nan)
        lower = np.full(len(high), np.nan)
        for i in range(period-1, len(high)):
            upper[i] = np.max(high[i-(period-1):i+1])
            lower[i] = np.min(low[i-(period-1):i+1])
        return upper, lower
    
    donch_upper, donch_lower = calculate_donchian(high, low, 20)
    
    # ADX calculation (14-period) for trend strength
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros(len(high))
        minus_dm = np.zeros(len(high))
        tr = np.zeros(len(high))
        
        for i in range(1, len(high)):
            plus_dm[i] = max(0, high[i] - high[i-1])
            minus_dm[i] = max(0, low[i-1] - low[i])
            if plus_dm[i] > minus_dm[i]:
                minus_dm[i] = 0
            elif minus_dm[i] > plus_dm[i]:
                plus_dm[i] = 0
            else:
                plus_dm[i] = 0
                minus_dm[i] = 0
            
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # ATR using Wilder's smoothing
        atr = np.zeros(len(high))
        atr[period] = np.nansum(tr[1:period+1]) / period
        for i in range(period+1, len(high)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        plus_di = np.zeros(len(high))
        minus_di = np.zeros(len(high))
        dx = np.zeros(len(high))
        
        for i in range(period, len(high)):
            if atr[i] != 0:
                plus_di[i] = 100 * (plus_dm[i] / atr[i])
                minus_di[i] = 100 * (minus_dm[i] / atr[i])
                if (plus_di[i] + minus_di[i]) != 0:
                    dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
        
        adx = np.zeros(len(high))
        adx[2*period-1] = np.nansum(dx[period:2*period]) / period
        for i in range(2*period, len(high)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx = calculate_adx(high, low, close, 14)
    
    # Volume confirmation: volume > 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Volatility regime filter: ATR ratio (current ATR / 50-period ATR) < 1.0 (avoid high volatility chop)
    def calculate_atr(high, low, close, period=14):
        tr = np.zeros(len(high))
        for i in range(1, len(high)):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        atr = np.zeros(len(high))
        atr[period] = np.nansum(tr[1:period+1]) / period
        for i in range(period+1, len(high)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        return atr
    
    atr = calculate_atr(high, low, close, 14)
    atr_ma = pd.Series(atr).rolling(window=50, min_periods=50).mean().values
    vol_regime = atr / atr_ma  # < 1.0 = low volatility, > 1.0 = high volatility
    
    # Higher timeframe trend filter: 12h EMA50
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if data not available
        if (np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]) or np.isnan(adx[i]) or 
            np.isnan(vol_ma[i]) or vol_ma[i] == 0 or np.isnan(ema_50_12h_aligned[i]) or
            np.isnan(vol_regime[i])):
            signals[i] = 0.0
            continue
        
        # Conditions
        vol_expansion = volume[i] > vol_ma[i]  # Volume above average
        low_vol_regime = vol_regime[i] < 1.0   # Avoid high volatility chop
        strong_trend = adx[i] > 25             # Trending market
        price_above_upper = close[i] > donch_upper[i]
        price_below_lower = close[i] < donch_lower[i]
        bullish_12h = ema_50_12h_aligned[i] > ema_50_12h_aligned[i-1] if i > 0 else False
        bearish_12h = ema_50_12h_aligned[i] < ema_50_12h_aligned[i-1] if i > 0 else False
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian lower OR trend weakens
            if close[i] < donch_lower[i] or adx[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above Donchian upper OR trend weakens
            if close[i] > donch_upper[i] or adx[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: Donchian breakout + volume + low vol regime + strong trend + bullish 12h trend
            if (price_above_upper and vol_expansion and low_vol_regime and strong_trend and 
                bullish_12h):
                position = 1
                signals[i] = 0.25
            # Short: Donchian breakdown + volume + low vol regime + strong trend + bearish 12h trend
            elif (price_below_lower and vol_expansion and low_vol_regime and strong_trend and 
                  bearish_12h):
                position = -1
                signals[i] = -0.25
    
    return signals