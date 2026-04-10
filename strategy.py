#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + ADX regime filter with 1w trend filter
# - Elder Ray: Bull Power = High - EMA13, Bear Power = EMA13 - Low
# - ADX > 25 indicates trending market
# - 1w EMA50 > EMA200 = bullish weekly trend bias for longs
# - 1w EMA50 < EMA200 = bearish weekly trend bias for shorts
# - Long when Bull Power > 0 AND ADX > 25 AND weekly bullish bias
# - Short when Bear Power > 0 AND ADX > 25 AND weekly bearish bias
# - Exit when power reverses sign OR ADX < 20 (regime change)
# - Uses discrete position sizing 0.25 to limit fee churn
# - Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years)
# - Elder Ray measures bull/bear strength relative to trend
# - ADX filter ensures we only trade in trending conditions
# - Weekly trend filter aligns with higher timeframe direction

name = "6h_1w_elder_ray_adx_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Pre-compute 6h OHLC
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Pre-compute 6h EMA13 for Elder Ray
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Pre-compute Elder Ray components
    bull_power = high - ema13  # Bull Power = High - EMA13
    bear_power = ema13 - low   # Bear Power = EMA13 - Low
    
    # Pre-compute 6h ADX (14-period)
    # True Range
    high_shift = np.roll(high, 1)
    low_shift = np.roll(low, 1)
    close_shift = np.roll(close, 1)
    high_shift[0] = high[0]
    low_shift[0] = low[0]
    close_shift[0] = close[0]
    
    tr1 = high - low
    tr2 = np.abs(high - close_shift)
    tr3 = np.abs(low - close_shift)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = high - high_shift
    down_move = low_shift - low
    up_move = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    down_move = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values using Wilder's smoothing (EMA with alpha=1/14)
    def wilders_smoothing(arr, period):
        result = np.zeros_like(arr)
        result[period-1] = np.mean(arr[1:period+1])  # First value
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    atr = wilders_smoothing(tr, 14)
    plus_dm = wilders_smoothing(up_move, 14)
    minus_dm = wilders_smoothing(down_move, 14)
    
    # DI values
    plus_di = 100 * plus_dm / atr
    minus_di = 100 * minus_dm / atr
    
    # DX and ADX
    dx = np.where((plus_di + minus_di) != 0, 
                  100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 
                  0)
    adx = wilders_smoothing(dx, 14)
    
    # Regime filters
    trending = adx > 25
    low_adx_exit = adx < 20  # Exit when trend weakens
    
    # Pre-compute 1w EMA50 and EMA200 for trend filter
    weekly_close = df_1w['close'].values
    weekly_close_s = pd.Series(weekly_close)
    ema50_1w = weekly_close_s.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema200_1w = weekly_close_s.ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Weekly trend bias
    weekly_bullish = ema50_1w > ema200_1w
    weekly_bearish = ema50_1w < ema200_1w
    
    # Align HTF indicators to 6h timeframe
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish)
    weekly_bearish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bearish)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(adx[i]) or np.isnan(weekly_bullish_aligned[i]) or 
            np.isnan(weekly_bearish_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: Bull Power > 0 AND ADX > 25 AND weekly bullish bias
            if (bull_power[i] > 0 and 
                trending[i] and 
                weekly_bullish_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short conditions: Bear Power > 0 AND ADX > 25 AND weekly bearish bias
            elif (bear_power[i] > 0 and 
                  trending[i] and 
                  weekly_bearish_aligned[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: power reverses sign OR ADX < 20 (regime change)
            exit_long = (position == 1 and (bull_power[i] <= 0 or low_adx_exit[i]))
            exit_short = (position == -1 and (bear_power[i] <= 0 or low_adx_exit[i]))
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals