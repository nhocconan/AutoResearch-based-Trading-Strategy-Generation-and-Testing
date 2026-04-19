#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Fractal breakout with weekly ADX trend filter and volume confirmation.
# Uses weekly ADX to filter trend strength (ADX > 25) and daily Williams Fractals for breakout signals.
# In strong trend (ADX > 25): buy when price breaks above recent bullish fractal, sell when breaks below bearish fractal.
# In weak trend (ADX <= 25): no trades to avoid whipsaws.
# Volume confirmation: volume > 1.5x 20-period average to avoid false breakouts.
# Target: 20-40 trades/year per symbol to stay within frequency limits.
name = "4h_WilliamsFractal_ADXTrend_Volume"
timeframe = "4h"
leverage = 1.0

def williams_fractal(high, low):
    """Calculate Williams Fractals: bearish (high) and bullish (low)"""
    n = len(high)
    bearish = np.full(n, np.nan)
    bullish = np.full(n, np.nan)
    
    for i in range(2, n-2):
        # Bearish fractal: high[i] is highest of 5 bars
        if (high[i] > high[i-1] and high[i] > high[i-2] and 
            high[i] > high[i+1] and high[i] > high[i+2]):
            bearish[i] = high[i]
        # Bullish fractal: low[i] is lowest of 5 bars
        if (low[i] < low[i-1] and low[i] < low[i-2] and 
            low[i] < low[i+1] and low[i] < low[i+2]):
            bullish[i] = low[i]
            
    return bearish, bullish

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)"""
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Directional Movement
    plus_dm = np.zeros_like(high)
    minus_dm = np.zeros_like(high)
    
    for i in range(1, len(high)):
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        else:
            plus_dm[i] = 0
            
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
        else:
            minus_dm[i] = 0
    
    # Smoothed TR, +DM, -DM using Wilder's smoothing
    def wilder_smooth(data, period):
        result = np.zeros_like(data)
        if len(data) < period:
            return result
        result[period-1] = np.nansum(data[:period])
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    tr_smooth = wilder_smooth(tr, period)
    plus_dm_smooth = wilder_smooth(plus_dm, period)
    minus_dm_smooth = wilder_smooth(minus_dm, period)
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / tr_smooth
    minus_di = 100 * minus_dm_smooth / tr_smooth
    
    # DX and ADX
    dx = np.zeros_like(close)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    dx = np.where((plus_di + minus_di) == 0, 0, dx)
    
    adx = wilder_smooth(dx, period)
    return adx

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for ADX trend filter
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly ADX (14-period)
    adx_1w = calculate_adx(high_1w, low_1w, close_1w, 14)
    
    # Get daily data for Williams Fractals
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate daily Williams Fractals
    bearish_fractal, bullish_fractal = williams_fractal(high_1d, low_1d)
    
    # Williams Fractals need 2 extra bars for confirmation (wait for 2 candles after fractal)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    # Align weekly ADX to 4h timeframe
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # Get 4h average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Ensure ADX (14*2+6) and Williams Fractals are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(adx_1w_aligned[i]) or np.isnan(bearish_fractal_aligned[i]) or 
            np.isnan(bullish_fractal_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        adx_val = adx_1w_aligned[i]
        bearish_fractal_val = bearish_fractal_aligned[i]
        bullish_fractal_val = bullish_fractal_aligned[i]
        vol_ma = vol_ma_20[i]
        vol = volume[i]
        
        # Volume confirmation threshold
        volume_confirmed = vol > 1.5 * vol_ma
        
        # Trend strength filter: only trade in strong trends (ADX > 25)
        strong_trend = adx_val > 25
        
        if position == 0:
            # Enter long when price breaks above bullish fractal in strong trend
            if strong_trend and volume_confirmed and not np.isnan(bullish_fractal_val):
                if price > bullish_fractal_val:
                    signals[i] = 0.25
                    position = 1
            # Enter short when price breaks below bearish fractal in strong trend
            elif strong_trend and volume_confirmed and not np.isnan(bearish_fractal_val):
                if price < bearish_fractal_val:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Exit long when price breaks below bearish fractal or ADX weakens
            if not strong_trend or (not np.isnan(bearish_fractal_val) and price < bearish_fractal_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short when price breaks above bullish fractal or ADX weakens
            if not strong_trend or (not np.isnan(bullish_fractal_val) and price > bullish_fractal_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals