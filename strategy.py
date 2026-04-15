#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Fractal + 1d Volume + 1d ADX Trend
# Williams Fractal identifies swing highs/lows for reversal signals
# Long when bullish fractal forms + volume spike + 1d ADX > 25 (trending)
# Short when bearish fractal forms + volume spike + 1d ADX > 25 (trending)
# Works in bull (fractal continuations) and bear (fractal reversals)
# Uses discrete sizing (0.25) to limit overtrading and fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1-day data for Williams Fractal and ADX
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams Fractal: 5-bar pattern
    # Bullish fractal: low[n-2] < low[n-1] and low[n] < low[n-1] and low[n+1] < low[n-1] and low[n+2] < low[n-1]
    # Bearish fractal: high[n-2] < high[n-1] and high[n] < high[n-1] and high[n+1] < high[n-1] and high[n+2] < high[n-1]
    n_1d = len(high_1d)
    bullish_fractal = np.zeros(n_1d, dtype=bool)
    bearish_fractal = np.zeros(n_1d, dtype=bool)
    
    for i in range(2, n_1d - 2):
        if (low_1d[i-2] < low_1d[i-1] and low_1d[i] < low_1d[i-1] and 
            low_1d[i+1] < low_1d[i-1] and low_1d[i+2] < low_1d[i-1]):
            bullish_fractal[i] = True
        if (high_1d[i-2] < high_1d[i-1] and high_1d[i] < high_1d[i-1] and 
            high_1d[i+1] < high_1d[i-1] and high_1d[i+2] < high_1d[i-1]):
            bearish_fractal[i] = True
    
    # Williams Fractal needs 2 extra bars for confirmation (pattern completes at bar i+2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal.astype(float), additional_delay_bars=2)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal.astype(float), additional_delay_bars=2)
    
    # 1-day ADX for trend strength
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            plus_dm[i] = max(0, high[i] - high[i-1])
            minus_dm[i] = max(0, low[i-1] - low[i])
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Smooth with Wilder's smoothing (alpha = 1/period)
        atr = np.zeros_like(tr)
        plus_dm_smooth = np.zeros_like(plus_dm)
        minus_dm_smooth = np.zeros_like(minus_dm)
        
        atr[period-1] = np.mean(tr[1:period]) if period > 1 else tr[0]
        plus_dm_smooth[period-1] = np.mean(plus_dm[1:period]) if period > 1 else plus_dm[0]
        minus_dm_smooth[period-1] = np.mean(minus_dm[1:period]) if period > 1 else minus_dm[0]
        
        for i in range(period, len(high)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
            plus_dm_smooth[i] = (plus_dm_smooth[i-1] * (period-1) + plus_dm[i]) / period
            minus_dm_smooth[i] = (minus_dm_smooth[i-1] * (period-1) + minus_dm[i]) / period
        
        plus_di = 100 * plus_dm_smooth / atr
        minus_di = 100 * minus_dm_smooth / atr
        dx = np.zeros_like(high)
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = np.zeros_like(high)
        
        # Smooth DX
        adx[2*period-2] = np.mean(dx[period-1:2*period-1]) if 2*period-1 > period-1 else dx[period-1]
        for i in range(2*period-1, len(high)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Volume confirmation: current > 2.0x median of last 24 bars (2 days)
    vol_median = pd.Series(volume).rolling(window=24, min_periods=1).median()
    vol_threshold = 2.0 * vol_median
    
    signals = np.zeros(n)
    
    for i in range(24, n):  # Start after warmup period
        # Skip if any required data is NaN
        if (np.isnan(bullish_fractal_aligned[i]) or np.isnan(bearish_fractal_aligned[i]) or 
            np.isnan(adx_1d_aligned[i]) or np.isnan(vol_threshold[i])):
            continue
        
        # Long: Bullish fractal + volume spike + ADX > 25
        if (bullish_fractal_aligned[i] > 0.5 and  # Fractal detected
            volume[i] > vol_threshold[i] and 
            adx_1d_aligned[i] > 25):
            signals[i] = 0.25
        
        # Short: Bearish fractal + volume spike + ADX > 25
        elif (bearish_fractal_aligned[i] > 0.5 and  # Fractal detected
              volume[i] > vol_threshold[i] and 
              adx_1d_aligned[i] > 25):
            signals[i] = -0.25
        
        # Exit: fractal signal disappears or ADX weakens
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and (bullish_fractal_aligned[i] <= 0.5 or adx_1d_aligned[i] <= 25)) or
               (signals[i-1] == -0.25 and (bearish_fractal_aligned[i] <= 0.5 or adx_1d_aligned[i] <= 25)))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "12h_WilliamsFractal_Volume_ADX"
timeframe = "12h"
leverage = 1.0