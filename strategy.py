#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h ADX trend strength with 4h EMA50 filter and volume confirmation
# Long when ADX(14) > 25 (trending) + price > EMA50(4h) + volume > 1.5x average
# Short when ADX(14) > 25 (trending) + price < EMA50(4h) + volume > 1.5x average
# Uses 4h EMA for trend direction to avoid 1h whipsaw, ADX for trend strength filter
# Volume confirms institutional participation. Targets 60-150 total trades over 4 years.

name = "1h_ADX_TrendStrength_4hEMA50_Volume"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data once for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA(50) for trend direction
    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Calculate ADX(14) for trend strength
    # TR = max(high-low, abs(high-close_prev), abs(low-close_prev))
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first value
    
    # +DM and -DM
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smooth TR, +DM, -DM using Wilder's smoothing (alpha = 1/period)
    def wilder_smooth(data, period):
        result = np.full_like(data, np.nan)
        alpha = 1.0 / period
        # First value is simple average
        if len(data) >= period:
            result[period-1] = np.nansum(data[:period]) / period
            for i in range(period, len(data)):
                result[i] = result[i-1] + alpha * (data[i] - result[i-1])
        return result
    
    atr = wilder_smooth(tr, 14)
    plus_di = 100 * wilder_smooth(plus_dm, 14) / atr
    minus_di = 100 * wilder_smooth(minus_dm, 14) / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilder_smooth(dx, 14)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(adx[i]) or np.isnan(ema50_4h_aligned[i]) or 
            np.isnan(plus_di[i]) or np.isnan(minus_di[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        adx_val = adx[i]
        ema50_4h_val = ema50_4h_aligned[i]
        plus_di_val = plus_di[i]
        minus_di_val = minus_di[i]
        vol_filt = volume_filter[i]
        
        if position == 0:
            # Enter long: ADX > 25 (trending) + +DI > -DI (bullish) + price > EMA50(4h) + volume filter
            if adx_val > 25 and plus_di_val > minus_di_val and close[i] > ema50_4h_val and vol_filt:
                signals[i] = 0.20
                position = 1
            # Enter short: ADX > 25 (trending) + -DI > +DI (bearish) + price < EMA50(4h) + volume filter
            elif adx_val > 25 and minus_di_val > plus_di_val and close[i] < ema50_4h_val and vol_filt:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: ADX < 20 (weak trend) OR -DI > +DI (bearish crossover)
            if adx_val < 20 or minus_di_val > plus_di_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: ADX < 20 (weak trend) OR +DI > -DI (bullish crossover)
            if adx_val < 20 or plus_di_val > minus_di_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals