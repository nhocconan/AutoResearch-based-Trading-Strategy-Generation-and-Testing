#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) + 1d ADX regime filter
# Long when Bull Power > 0 AND ADX > 25 (trending market) AND EMA(13) > EMA(34) (uptrend bias)
# Short when Bear Power < 0 AND ADX > 25 (trending market) AND EMA(13) < EMA(34) (downtrend bias)
# Exit when Elder Power reverses sign OR ADX < 20 (range market)
# Uses discrete sizing 0.25 to balance return and fee drag
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe
# Elder Ray measures bull/bear strength relative to EMA13; ADX filters for trending regimes
# Works in both bull (strong uptrends) and bear (strong downtrends) markets by requiring ADX>25

name = "6h_ElderRay_1dADX25_EMA1334_Trend"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate EMA13 and EMA34 for Elder Ray
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    ema34 = close_s.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Get 1d data ONCE before loop for ADX and EMA filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX (14-period)
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
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Wilder's smoothing
        atr = np.zeros_like(high)
        atr[period] = np.mean(tr[1:period+1])
        for i in range(period+1, len(high)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        plus_di = 100 * (np.zeros_like(high))
        minus_di = 100 * (np.zeros_like(high))
        plus_dm_smooth = np.zeros_like(high)
        minus_dm_smooth = np.zeros_like(high)
        
        plus_dm_smooth[period] = np.mean(plus_dm[1:period+1])
        minus_dm_smooth[period] = np.mean(minus_dm[1:period+1])
        
        for i in range(period+1, len(high)):
            plus_dm_smooth[i] = (plus_dm_smooth[i-1] * (period-1) + plus_dm[i]) / period
            minus_dm_smooth[i] = (minus_dm_smooth[i-1] * (period-1) + minus_dm[i]) / period
            plus_di[i] = 100 * plus_dm_smooth[i] / atr[i]
            minus_di[i] = 100 * minus_dm_smooth[i] / atr[i]
        
        dx = np.zeros_like(high)
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        
        adx = np.zeros_like(high)
        adx[2*period-1] = np.mean(dx[period:2*period])
        for i in range(2*period, len(high)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    
    # Calculate 1d EMA34 for trend bias
    close_1d_s = pd.Series(close_1d)
    ema34_1d = close_1d_s.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align HTF indicators to 6h timeframe (wait for completed HTF bar)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(ema13[i]) or np.isnan(ema34[i]) or 
            np.isnan(adx_1d_aligned[i]) or np.isnan(ema34_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Entry conditions: Elder Ray + ADX > 25 + EMA13/EMA34 alignment
            # Long: Bull Power > 0 AND ADX > 25 AND EMA13 > EMA34 (uptrend bias)
            if bull_power[i] > 0 and adx_1d_aligned[i] > 25 and ema13[i] > ema34[i]:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 AND ADX > 25 AND EMA13 < EMA34 (downtrend bias)
            elif bear_power[i] < 0 and adx_1d_aligned[i] > 25 and ema13[i] < ema34[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Bear Power < 0 OR ADX < 20 (range) OR EMA13 < EMA34
            if bear_power[i] < 0 or adx_1d_aligned[i] < 20 or ema13[i] < ema34[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bull Power > 0 OR ADX < 20 (range) OR EMA13 > EMA34
            if bull_power[i] > 0 or adx_1d_aligned[i] < 20 or ema13[i] > ema34[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals