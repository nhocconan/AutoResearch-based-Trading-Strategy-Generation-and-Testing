#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band squeeze breakout with 1d ADX trend filter and volume confirmation.
# Long when BB width < 20th percentile (squeeze) AND price breaks above upper band AND 1d ADX > 25 (trending) AND volume > 1.5x 20-period average.
# Short when BB width < 20th percentile (squeeze) AND price breaks below lower band AND 1d ADX > 25 (trending) AND volume > 1.5x 20-period average.
# Uses discrete position size 0.25. BB squeeze identifies low volatility primed for breakout,
# 1d ADX ensures alignment with higher timeframe trend strength, volume confirms breakout validity.
# Designed to work in both bull (buy breakouts) and bear (sell breakdowns) markets.
# Target: 80-160 trades over 4 years (20-40/year) to balance opportunity and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 6h Indicators: Bollinger Bands (20, 2) ===
    close_series = pd.Series(close)
    bb_middle = close_series.rolling(window=20, min_periods=20).mean().values
    bb_std = close_series.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    bb_width = bb_upper - bb_lower
    
    # BB width percentile (20-period lookback for regime)
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=20, min_periods=20).rank(pct=True).values
    
    # === 6h Indicators: Volume Spike (volume > 1.5x 20-period average) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Get 1d data once before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough for ADX calculation
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # === 1d Indicators: ADX (14-period) for trend strength ===
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            plus_dm[i] = max(0, high[i] - high[i-1])
            minus_dm[i] = max(0, low[i-1] - low[i])
            if plus_dm[i] < minus_dm[i]:
                plus_dm[i] = 0
            if minus_dm[i] < plus_dm[i]:
                minus_dm[i] = 0
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Smooth with Wilder's smoothing (alpha = 1/period)
        def WilderSmoothing(data, period):
            smoothed = np.zeros_like(data)
            smoothed[period-1] = np.mean(data[:period])
            for i in range(period, len(data)):
                smoothed[i] = (smoothed[i-1] * (period-1) + data[i]) / period
            return smoothed
        
        smoothed_tr = WilderSmoothing(tr, period)
        smoothed_plus_dm = WilderSmoothing(plus_dm, period)
        smoothed_minus_dm = WilderSmoothing(minus_dm, period)
        
        plus_di = 100 * smoothed_plus_dm / smoothed_tr
        minus_di = 100 * smoothed_minus_dm / smoothed_tr
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = WilderSmoothing(dx, period)
        
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    
    # Align 1d ADX to 6h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 20 periods for BB, 34 for ADX smoothing)
    warmup = 40
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(bb_width_percentile[i]) or np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or
            np.isnan(adx_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        bb_width_pct = bb_width_percentile[i]
        bb_up = bb_upper[i]
        bb_low = bb_lower[i]
        adx_val = adx_1d_aligned[i]
        vol_spike = volume_spike[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price crosses below middle band or BB squeeze ends or ADX weakens
            if price < bb_middle[i] or bb_width_pct > 0.3 or adx_val < 20:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price crosses above middle band or BB squeeze ends or ADX weakens
            if price > bb_middle[i] or bb_width_pct > 0.3 or adx_val < 20:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: BB squeeze (width < 20th percentile) AND price breaks above upper band AND 1d ADX > 25 (trending) AND volume spike
            if bb_width_pct < 0.2 and price > bb_up and adx_val > 25 and vol_spike:
                signals[i] = 0.25
                position = 1
            
            # SHORT: BB squeeze (width < 20th percentile) AND price breaks below lower band AND 1d ADX > 25 (trending) AND volume spike
            elif bb_width_pct < 0.2 and price < bb_low and adx_val > 25 and vol_spike:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "6h_BB_Squeeze_Breakout_1dADX_VolumeSpike_V1"
timeframe = "6h"
leverage = 1.0