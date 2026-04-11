#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + ADX regime filter
# - Bull Power = High - EMA(13), Bear Power = EMA(13) - Low
# - Long: Bull Power > 0 AND Bear Power < 0 AND ADX > 25 (strong trend)
# - Short: Bear Power > 0 AND Bull Power < 0 AND ADX > 25 (strong trend)
# - Exit: ADX < 20 (trend weakening) or power divergence
# - Uses 1d EMA for Elder Ray calculation (more stable) aligned to 6h
# - ADX calculated on 6h for trend strength
# - Works in both bull and bear markets by only trading strong trends
# - Target: 12-37 trades/year (50-150 total over 4 years) to stay within fee drag limits

name = "6h_1d_elder_ray_adx_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1d data ONCE before loop for Elder Ray calculation (MTF rule compliance)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return signals
    
    # Pre-compute 1d EMA(13) for Elder Ray
    close_1d = df_1d['close'].values
    ema_13_1d = pd.Series(close_1d).ewm(span=13, min_periods=13, adjust=False).mean().values
    
    # Align 1d EMA to 6h timeframe
    ema_13_aligned = align_htf_to_ltf(prices, df_1d, ema_13_1d)
    
    # Pre-compute 6h ADX (14-period)
    # ADX requires +DI, -DI, and TR
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    
    for i in range(1, n):
        plus_dm[i] = max(0, high[i] - high[i-1]) if (high[i] - high[i-1]) > (low[i-1] - low[i]) else 0
        minus_dm[i] = max(0, low[i-1] - low[i]) if (low[i-1] - low[i]) > (high[i] - high[i-1]) else 0
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Smooth with Wilder's smoothing (alpha = 1/period)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(data[:period])
        # Subsequent values: Wilder's smoothing
        for i in range(period, len(data)):
            if not np.isnan(result[i-1]):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    period_adx = 14
    tr_smooth = wilders_smoothing(tr, period_adx)
    plus_dm_smooth = wilders_smoothing(plus_dm, period_adx)
    minus_dm_smooth = wilders_smoothing(minus_dm, period_adx)
    
    # Avoid division by zero
    plus_di = np.where(tr_smooth != 0, (plus_dm_smooth / tr_smooth) * 100, 0)
    minus_di = np.where(tr_smooth != 0, (minus_dm_smooth / tr_smooth) * 100, 0)
    
    dx = np.where((plus_di + minus_di) != 0, 
                  np.abs((plus_di - minus_di) / (plus_di + minus_di)) * 100, 
                  0)
    adx = wilders_smoothing(dx, period_adx)
    
    # Pre-compute 6h volume confirmation (20-period average)
    volume = prices['volume'].values
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(ema_13_aligned[i]) or np.isnan(adx[i]) or 
            np.isnan(volume_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        high_price = high[i]
        low_price = low[i]
        volume_current = volume[i]
        
        # Elder Ray components
        bull_power = high_price - ema_13_aligned[i]
        bear_power = ema_13_aligned[i] - low_price
        
        # ADX trend strength
        adx_value = adx[i]
        
        # Volume confirmation: current volume > 1.3x 20-period average
        vol_confirm = volume_current > 1.3 * volume_sma_20[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Bull Power > 0 AND Bear Power < 0 AND ADX > 25 (strong uptrend)
        if bull_power > 0 and bear_power < 0 and adx_value > 25 and vol_confirm:
            enter_long = True
        
        # Short: Bear Power > 0 AND Bull Power < 0 AND ADX > 25 (strong downtrend)
        if bear_power > 0 and bull_power < 0 and adx_value > 25 and vol_confirm:
            enter_short = True
        
        # Exit conditions: trend weakening or power divergence
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if ADX < 20 (trend weakening) or Bear Power becomes positive
            exit_long = adx_value < 20 or bear_power > 0
        elif position == -1:
            # Exit short if ADX < 20 (trend weakening) or Bull Power becomes positive
            exit_short = adx_value < 20 or bull_power > 0
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals