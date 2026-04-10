#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 1d ADX regime filter for BTC/ETH
# - Elder Ray: Bull Power = High - EMA13, Bear Power = EMA13 - Low
# - Regime filter: ADX(14) > 25 = trending, ADX < 20 = ranging (hysteresis)
# - Long: Bull Power > 0 AND Bear Power < 0 AND ADX trending AND 6h close > 6h EMA50
# - Short: Bear Power > 0 AND Bull Power < 0 AND ADX trending AND 6h close < 6h EMA50
# - Exit: Power signals reverse OR ADX enters ranging regime
# - Uses 1d timeframe for ADX/EMA13 to avoid 6h noise, 6h for entry timing
# - Position sizing: 0.25 discrete level to minimize fee drag
# - Target: 12-37 trades/year on 6h timeframe to stay within fee drag limits

name = "6h_1d_elder_ray_adx_regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Calculate 1d EMA13 for Elder Ray
    close_1d = df_1d['close'].values
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate 1d ADX for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    def ma_smoothing(data, period):
        result = np.zeros_like(data)
        result[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr = ma_smoothing(tr, 14)
    dm_plus_smooth = ma_smoothing(dm_plus, 14)
    dm_minus_smooth = ma_smoothing(dm_minus, 14)
    
    # Avoid division by zero
    atr_safe = np.where(atr == 0, 1e-10, atr)
    di_plus = 100 * dm_plus_smooth / atr_safe
    di_minus = 100 * dm_minus_smooth / atr_safe
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = ma_smoothing(dx, 14)
    
    # Align 1d indicators to 6h timeframe
    ema_13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_13_1d)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate 6h EMA50 for trend confirmation
    ema_50_6h = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    for i in range(50, n):  # Start after warmup for indicators
        # Skip if any required data is invalid
        if (np.isnan(ema_13_1d_aligned[i]) or np.isnan(adx_aligned[i]) or
            np.isnan(ema_50_6h[i])):
            signals[i] = 0.0
            continue
        
        # Elder Ray calculation using current bar's high/low and aligned EMA13
        bull_power = high[i] - ema_13_1d_aligned[i]
        bear_power = ema_13_1d_aligned[i] - low[i]
        
        # Regime filter with hysteresis: ADX > 25 = trending, ADX < 20 = ranging
        adx_now = adx_aligned[i]
        adx_prev = adx_aligned[i-1] if i > 0 else 25
        
        # Hysteresis logic
        if i == 50:
            adx_state = 1 if adx_now > 25 else 0  # 1=trending, 0=ranging
        else:
            if adx_state == 1:  # Currently trending
                adx_state = 0 if adx_now < 20 else 1
            else:  # Currently ranging
                adx_state = 1 if adx_now > 25 else 0
        
        is_trending = (adx_state == 1)
        
        # 6h EMA50 for additional trend filter
        ema_50_now = ema_50_6h[i]
        ema_50_prev = ema_50_6h[i-1] if i > 0 else close[i]
        ema_50_trend_up = close[i] > ema_50_now
        ema_50_trend_down = close[i] < ema_50_now
        
        # Entry conditions
        long_entry = (bull_power > 0 and bear_power < 0 and 
                     is_trending and ema_50_trend_up)
        short_entry = (bear_power > 0 and bull_power < 0 and 
                      is_trending and ema_50_trend_down)
        
        # Exit conditions: power reversal or loss of trending regime
        long_exit = (bull_power < 0 or bear_power > 0 or not is_trending)
        short_exit = (bear_power < 0 or bull_power > 0 or not is_trending)
        
        if position == 0:  # Flat - look for entry
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        elif position == 1:  # Long position - look for exit
            if long_exit:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        else:  # position == -1 (Short position) - look for exit
            if short_exit:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
    
    return signals