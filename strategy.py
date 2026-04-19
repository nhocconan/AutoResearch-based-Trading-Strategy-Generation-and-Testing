#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA trend with weekly ADX regime filter and volume confirmation.
# Long when KAMA indicates uptrend, weekly ADX > 25 (trending), and volume > 1.5x average.
# Short when KAMA indicates downtrend, weekly ADX > 25, and volume > 1.5x average.
# Uses weekly ADX to filter for trending markets only, avoiding whipsaw in sideways markets.
# KAMA adapts to market noise, reducing false signals in choppy conditions.
# Volume confirmation ensures moves have institutional participation.
# Target: 10-25 trades/year per symbol (~40-100 total over 4 years).
name = "1d_KAMA_ADX25_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for ADX calculation
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate ADX on weekly data (14-period)
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # First period
        
        # Directional Movement
        up_move = high - np.roll(high, 1)
        down_move = np.roll(low, 1) - low
        up_move[0] = 0
        down_move[0] = 0
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Smoothed values
        def smooth_series(values, period):
            smoothed = np.full_like(values, np.nan, dtype=float)
            if len(values) < period:
                return smoothed
            # Initial value: simple average
            smoothed[period-1] = np.nansum(values[:period]) / period
            # Wilder smoothing
            for i in range(period, len(values)):
                if not np.isnan(smoothed[i-1]):
                    smoothed[i] = (smoothed[i-1] * (period-1) + values[i]) / period
            return smoothed
        
        atr = smooth_series(tr, period)
        plus_di = 100 * smooth_series(plus_dm, period) / atr
        minus_di = 100 * smooth_series(minus_dm, period) / atr
        dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
        adx = smooth_series(dx, period)
        return adx
    
    adx_1w = calculate_adx(high_1w, low_1w, close_1w, 14)
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # Calculate KAMA on daily close
    def kama(close, er_length=10, fast_ema=2, slow_ema=30):
        change = np.abs(close - np.roll(close, er_length))
        change[0] = np.abs(close[0] - close[0])  # Avoid NaN
        
        volatility = np.sum(np.abs(np.diff(close, prepend=close[0]))[:er_length]) if len(close) >= er_length else 0
        for i in range(er_length, len(close)):
            volatility = volatility - np.abs(close[i-er_length] - close[i-er_length+1]) + np.abs(close[i-1] - close[i])
        
        # Avoid division by zero
        er = np.where(volatility != 0, change / volatility, 0)
        sc = np.power(er * (2/(fast_ema+1) - 2/(slow_ema+1)) + 2/(slow_ema+1), 2)
        
        kama_vals = np.full_like(close, np.nan, dtype=float)
        kama_vals[0] = close[0]
        for i in range(1, len(close)):
            if not np.isnan(kama_vals[i-1]):
                kama_vals[i] = kama_vals[i-1] + sc[i] * (close[i] - kama_vals[i-1])
        return kama_vals
    
    kama_vals = kama(close, 10, 2, 30)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 30)  # Need volume MA and KAMA data
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(adx_1w_aligned[i]) or np.isnan(kama_vals[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        adx_val = adx_1w_aligned[i]
        kama_val = kama_vals[i]
        vol_ma = vol_ma_20[i]
        vol = volume[i]
        
        # Conditions
        trending_market = adx_val > 25
        volume_confirmed = vol > 1.5 * vol_ma
        kama_bullish = price > kama_val
        kama_bearish = price < kama_val
        
        if position == 0:
            # Enter long: KAMA bullish AND trending market AND volume confirmation
            if kama_bullish and trending_market and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Enter short: KAMA bearish AND trending market AND volume confirmation
            elif kama_bearish and trending_market and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long when KAMA turns bearish OR loses trend
            if not kama_bullish or not trending_market:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short when KAMA turns bullish OR loses trend
            if not kama_bearish or not trending_market:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals