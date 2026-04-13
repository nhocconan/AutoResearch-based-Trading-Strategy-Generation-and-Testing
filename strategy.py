#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Williams Fractal breakouts with volume confirmation and ADX trend filter.
# Long: Price breaks above bearish fractal (resistance) + volume > 1.5x avg + ADX > 25.
# Short: Price breaks below bullish fractal (support) + volume > 1.5x avg + ADX > 25.
# Williams Fractals identify key support/resistance levels. Fractal breaks with volume and trend strength
# indicate genuine breakouts. Works in both bull/bear markets by trading breakouts in direction of trend.
# Target: 20-50 trades/year (80-200 over 4 years) to avoid fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data for Williams Fractals
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Williams Fractals
    # Bearish fractal: high[n-2] is highest of high[n-4:n-1] and high[n+1:n+2]
    # Bullish fractal: low[n-2] is lowest of low[n-4:n-1] and low[n+1:n+2]
    bearish_fractal = np.full(len(high_1d), np.nan)
    bullish_fractal = np.full(len(low_1d), np.nan)
    
    for i in range(2, len(high_1d) - 2):
        # Bearish fractal (resistance)
        left_high = high_1d[i-2:i]  # i-2, i-1
        right_high = high_1d[i+1:i+3]  # i+1, i+2
        if high_1d[i] >= np.max(left_high) and high_1d[i] >= np.max(right_high):
            bearish_fractal[i] = high_1d[i]
        
        # Bullish fractal (support)
        left_low = low_1d[i-2:i]  # i-2, i-1
        right_low = low_1d[i+1:i+3]  # i+1, i+2
        if low_1d[i] <= np.min(left_low) and low_1d[i] <= np.min(right_low):
            bullish_fractal[i] = low_1d[i]
    
    # Williams Fractals need 2 extra bars for confirmation (fractal forms after 2 bars)
    bearish_fractal_confirmed = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_confirmed = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    # ADX (14) for trend strength
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
        
        # Smooth with Wilder's smoothing (alpha = 1/period)
        atr = np.zeros_like(high)
        plus_di = np.zeros_like(high)
        minus_di = np.zeros_like(high)
        dx = np.zeros_like(high)
        
        atr[period] = np.mean(tr[1:period+1])
        plus_dm_smooth = np.zeros_like(high)
        minus_dm_smooth = np.zeros_like(high)
        
        plus_dm_smooth[period] = np.mean(plus_dm[1:period+1])
        minus_dm_smooth[period] = np.mean(minus_dm[1:period+1])
        
        for i in range(period + 1, len(high)):
            atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
            plus_dm_smooth[i] = (plus_dm_smooth[i-1] * (period - 1) + plus_dm[i]) / period
            minus_dm_smooth[i] = (minus_dm_smooth[i-1] * (period - 1) + minus_dm[i]) / period
            
            if atr[i] != 0:
                plus_di[i] = 100 * plus_dm_smooth[i] / atr[i]
                minus_di[i] = 100 * minus_dm_smooth[i] / atr[i]
                dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
        
        # ADX is smoothed DX
        adx = np.zeros_like(high)
        adx[2*period] = np.mean(dx[period+1:2*period+1])
        for i in range(2*period + 1, len(high)):
            adx[i] = (adx[i-1] * (period - 1) + dx[i]) / period
        
        return adx
    
    adx = calculate_adx(high, low, close, 14)
    
    # Average volume (20-period) for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(30, n):  # Start after ADX warmup
        # Skip if any required data is not ready
        if (np.isnan(bearish_fractal_confirmed[i]) or np.isnan(bullish_fractal_confirmed[i]) or 
            np.isnan(adx[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        bearish = bearish_fractal_confirmed[i]
        bullish = bullish_fractal_confirmed[i]
        trend_strength = adx[i]
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirm = vol > 1.5 * avg_vol
        # Trend filter: ADX > 25 indicates strong trend
        trend_filter = trend_strength > 25
        
        if position == 0:
            # Long: price breaks above bearish fractal (resistance) + volume + trend
            if (price > bearish and volume_confirm and trend_filter):
                position = 1
                signals[i] = position_size
            # Short: price breaks below bullish fractal (support) + volume + trend
            elif (price < bullish and volume_confirm and trend_filter):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price closes below bullish fractal (support)
            if price < bullish:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price closes above bearish fractal (resistance)
            if price > bearish:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_Williams_Fractal_Breakout_ADX_Volume"
timeframe = "4h"
leverage = 1.0