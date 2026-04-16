#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h/1d confluence for direction and 1h for precise entry timing.
# Long when: 4h EMA21 > 4h EMA50 (uptrend) AND 1d close > 1d EMA50 (bullish bias) AND
#            1h price breaks above 1h Donchian(20) upper with volume > 1.3x 20-period median volume.
# Short when: 4h EMA21 < 4h EMA50 (downtrend) AND 1d close < 1d EMA50 (bearish bias) AND
#             1h price breaks below 1h Donchian(20) lower with volume > 1.3x 20-period median volume.
# Uses discrete position size 0.20. Exits when price returns to 1h Donchian middle or when
# 4h trend reverses (EMA21/EMA50 cross). Session filter: 08-20 UTC to reduce noise.
# Target: 15-37 trades/year (60-150 over 4 years) to minimize fee drag. Works in both bull
# and bear markets by using HTF trend alignment + volume confirmation + session filter.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC) - open_time is already datetime64[ms]
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data once before loop for trend
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 60:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # === 4h Indicators: EMA21 and EMA50 for trend ===
    ema21_4h = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # 4h trend: EMA21 > EMA50 = uptrend, EMA21 < EMA50 = downtrend
    ema21_gt_ema50_4h = ema21_4h > ema50_4h
    ema21_lt_ema50_4h = ema21_4h < ema50_4h
    
    # Align 4h trend to 1h timeframe
    ema21_gt_ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema21_gt_ema50_4h)
    ema21_lt_ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema21_lt_ema50_4h)
    
    # Get 1d data once before loop for bias filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # === 1d Indicators: EMA50 for bias ===
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    close_gt_ema50_1d = close_1d > ema50_1d  # bullish bias
    close_lt_ema50_1d = close_1d < ema50_1d  # bearish bias
    
    # Align 1d bias to 1h timeframe
    close_gt_ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, close_gt_ema50_1d)
    close_lt_ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, close_lt_ema50_1d)
    
    # === 1h Indicators: Donchian Channel (20) and Volume Median ===
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_upper = highest_20
    donchian_lower = lowest_20
    donchian_middle = (highest_20 + lowest_20) / 2.0
    
    vol_median_20 = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(50, 20)  # 4h EMA50 needs 50, 1h Donchian/volume needs 20
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            position = 0
            continue
            
        # Skip if any required data is NaN
        if (np.isnan(ema21_gt_ema50_4h_aligned[i]) or np.isnan(ema21_lt_ema50_4h_aligned[i]) or
            np.isnan(close_gt_ema50_1d_aligned[i]) or np.isnan(close_lt_ema50_1d_aligned[i]) or
            np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or np.isnan(donchian_middle[i]) or
            np.isnan(vol_median_20[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        upper = donchian_upper[i]
        lower = donchian_lower[i]
        middle = donchian_middle[i]
        vol_median = vol_median_20[i]
        
        # Volume spike filter: current volume > 1.3x median volume
        volume_spike = volume[i] > (vol_median * 1.3)
        
        # Get aligned HTF values
        uptrend_4h = ema21_gt_ema50_4h_aligned[i]
        downtrend_4h = ema21_lt_ema50_4h_aligned[i]
        bullish_bias_1d = close_gt_ema50_1d_aligned[i]
        bearish_bias_1d = close_lt_ema50_1d_aligned[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit when price returns to middle OR 4h trend turns down
            if (price <= middle) or (not uptrend_4h):
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit when price returns to middle OR 4h trend turns up
            if (price >= middle) or (not downtrend_4h):
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: 4h uptrend + 1d bullish bias + price breaks above upper Donchian + volume spike
            if uptrend_4h and bullish_bias_1d and (price > upper) and volume_spike:
                signals[i] = 0.20
                position = 1
            
            # SHORT: 4h downtrend + 1d bearish bias + price breaks below lower Donchian + volume spike
            elif downtrend_4h and bearish_bias_1d and (price < lower) and volume_spike:
                signals[i] = -0.20
                position = -1
        
        else:
            signals[i] = position * 0.20  # maintain position
    
    return signals

name = "1h_4hEMA21_50_1dEMA50_1hDonchian20_VolumeSpike1.3x_SessionFilter_V1"
timeframe = "1h"
leverage = 1.0