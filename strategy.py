#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 1d Regime Filter
# - Uses 1d HTF for regime detection (ADX > 25 = trending, < 20 = ranging)
# - 6h Elder Ray (Bull Power = High - EMA13, Bear Power = Low - EMA13)
# - Long when Bull Power > 0 AND Bear Power rising (from negative) in trending up regime
# - Short when Bear Power < 0 AND Bull Power falling (from positive) in trending down regime
# - Regime filter avoids whipsaws in ranging markets
# - Fixed position size 0.25 to control drawdown
# - Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years)

name = "6h_1d_elder_ray_regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d indicators for regime
    # ADX calculation
    plus_dm = np.zeros(len(high_1d))
    minus_dm = np.zeros(len(high_1d))
    tr = np.zeros(len(high_1d))
    
    for i in range(1, len(high_1d)):
        plus_dm[i] = max(0, high_1d[i] - high_1d[i-1])
        minus_dm[i] = max(0, low_1d[i-1] - low_1d[i])
        tr[i] = max(high_1d[i] - low_1d[i], 
                   abs(high_1d[i] - close_1d[i-1]), 
                   abs(low_1d[i] - close_1d[i-1]))
    
    # Smooth with Wilder's smoothing (alpha = 1/period)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        result[period-1] = np.nansum(data[1:period])
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    period_adx = 14
    plus_di_1d = 100 * wilders_smoothing(plus_dm, period_adx) / wilders_smoothing(tr, period_adx)
    minus_di_1d = 100 * wilders_smoothing(minus_dm, period_adx) / wilders_smoothing(tr, period_adx)
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)
    adx_1d = wilders_smoothing(dx_1d, period_adx)
    
    # Regime: ADX > 25 = trending, ADX < 20 = ranging
    # Add hysteresis to avoid rapid regime changes
    trending = adx_1d > 25
    ranging = adx_1d < 20
    
    # Calculate 6h EMA13 for Elder Ray
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema_13  # High - EMA
    bear_power = low - ema_13   # Low - EMA
    
    # Align 1d regime data to 6h timeframe (wait for completed 1d bar)
    trending_aligned = align_htf_to_ltf(prices, df_1d, trending.astype(float))
    ranging_aligned = align_htf_to_ltf(prices, df_1d, ranging.astype(float))
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    position_size = 0.25
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(trending_aligned[i]) or np.isnan(ranging_aligned[i]) or
            np.isnan(ema_13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i])):
            signals[i] = 0.0
            continue
        
        # Regime check: only trade in trending markets
        is_trending = trending_aligned[i] > 0.5
        is_ranging = ranging_aligned[i] > 0.5
        
        if position == 1:  # Long position
            # Exit when bear power becomes positive (momentum fading) OR regime changes to ranging
            if bear_power[i] > 0 or is_ranging:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
                
        elif position == -1:  # Short position
            # Exit when bull power becomes negative (momentum fading) OR regime changes to ranging
            if bull_power[i] < 0 or is_ranging:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
        else:  # Flat
            # Only enter in trending regime
            if is_trending:
                # Long: Bull power positive AND bear power rising from negative (momentum building)
                if bull_power[i] > 0 and bear_power[i] < 0 and bear_power[i] > bear_power[i-1]:
                    position = 1
                    signals[i] = position_size
                # Short: Bear power negative AND bull power falling from positive (momentum building)
                elif bear_power[i] < 0 and bull_power[i] > 0 and bull_power[i] < bull_power[i-1]:
                    position = -1
                    signals[i] = -position_size
    
    return signals