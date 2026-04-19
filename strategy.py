#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h ADX trend filter + price action reversal at 1d support/resistance
# - Use 1d ADX(14) > 25 to identify trending markets
# - In uptrend (ADX > 25 and +DI > -DI): buy on pullbacks to 1d EMA(20)
# - In downtrend (ADX > 25 and -DI > +DI): sell on rallies to 1d EMA(20)
# - Use 12h price action: enter only when 12h close crosses above/below 12h EMA(20) in trend direction
# - Exit when trend weakens (ADX < 20) or opposite signal
# - Designed to capture trend continuation moves while avoiding choppy markets
# - Target: 15-30 trades/year to minimize fee drift

name = "12h_ADX_Trend_Pullback_EMA20_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for ADX and EMA
    df_1d = get_htf_data(prices, '1d')
    
    # 1d ADX calculation
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values (Wilder's smoothing)
    def wilders_smoothing(values, period):
        smoothed = np.zeros_like(values)
        smoothed[period-1] = np.nansum(values[:period])
        for i in range(period, len(values)):
            smoothed[i] = smoothed[i-1] - (smoothed[i-1] / period) + values[i]
        return smoothed
    
    tr_14 = wilders_smoothing(tr, 14)
    dm_plus_14 = wilders_smoothing(dm_plus, 14)
    dm_minus_14 = wilders_smoothing(dm_minus, 14)
    
    # DI values
    plus_di_14 = 100 * dm_plus_14 / (tr_14 + 1e-10)
    minus_di_14 = 100 * dm_minus_14 / (tr_14 + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14 + 1e-10)
    adx = wilders_smoothing(dx, 14)
    
    # 1d EMA(20) for pullback levels
    ema_20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align HTF indicators to 12h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx)
    plus_di_1d_aligned = align_htf_to_ltf(prices, df_1d, plus_di_14)
    minus_di_1d_aligned = align_htf_to_ltf(prices, df_1d, minus_di_14)
    ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    
    # 12h EMA(20) for entry timing
    ema_20_12h = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(plus_di_1d_aligned[i]) or 
            np.isnan(minus_di_1d_aligned[i]) or np.isnan(ema_20_1d_aligned[i]) or
            np.isnan(ema_20_12h[i])):
            signals[i] = 0.0
            continue
        
        # Trend detection
        is_uptrend = adx_1d_aligned[i] > 25 and plus_di_1d_aligned[i] > minus_di_1d_aligned[i]
        is_downtrend = adx_1d_aligned[i] > 25 and minus_di_1d_aligned[i] > plus_di_1d_aligned[i]
        trend_weakening = adx_1d_aligned[i] < 20
        
        if position == 0:
            # Look for long entry: uptrend + pullback to 1d EMA + 12h EMA cross up
            if (is_uptrend and 
                low[i] <= ema_20_1d_aligned[i] and  # Pullback to 1d support
                close[i] > ema_20_12h[i]):           # 12h momentum confirmation
                signals[i] = 0.25
                position = 1
            # Look for short entry: downtrend + rally to 1d EMA + 12h EMA cross down
            elif (is_downtrend and 
                  high[i] >= ema_20_1d_aligned[i] and  # Rally to 1d resistance
                  close[i] < ema_20_12h[i]):           # 12h momentum confirmation
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit on trend weakening or opposite signal
            if trend_weakening or (close[i] < ema_20_12h[i] and high[i] >= ema_20_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit on trend weakening or opposite signal
            if trend_weakening or (close[i] > ema_20_12h[i] and low[i] <= ema_20_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals