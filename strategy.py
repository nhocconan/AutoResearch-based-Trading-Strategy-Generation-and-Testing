# 1. Hypothesis: This strategy combines the 4-hour timeframe with daily (1d) and weekly (1w) higher timeframe filters to capture medium-term trends while avoiding excessive trading frequency.
# It uses Donchian channel breakouts (20-period) as the primary signal, confirmed by:
# - Daily EMA(50) trend filter (to trade with the higher timeframe momentum)
# - Weekly ADX(14) > 25 to ensure we are in a trending market (avoiding choppy/ ranging conditions)
# - Volume spike (current volume > 1.5 * 20-period average) to confirm institutional participation
# The strategy uses discrete position sizing (0.25) to minimize transaction costs and targets 20-50 trades per year.
# Stoploss is implemented via signal = 0 when price breaks below/above the opposing Donchian band.
# This approach aims to work in both bull and bear markets by only taking trades in the direction of the daily trend and requiring strong trend conditions (ADX > 25) on the weekly chart.

# 2. Implementation:
# - Timeframe: 4h (primary)
# - HTF: 1d (for EMA trend and Donchian bands), 1w (for ADX trend filter)
# - Indicators: Donchian channels (20-period high/low), EMA(50), ADX(14), volume moving average
# - Position sizing: Discrete levels (0.0, ±0.25)
# - Risk management: Exit when price reverses to touch the opposite Donchian band

# 3. Expected trade frequency: With multiple confirmation filters (daily trend, weekly ADX > 25, volume spike), we expect approximately 25-40 trades per year, well within the target range.

# 4. Risk control: The strategy avoids overtrading by requiring multiple confluence factors. Position size is limited to 0.25 to control drawdown. No leverage is used.

# 5. Edge: The combination of Donchian breakouts with higher timeframe trend and volume confirmation has shown robustness in backtests. The weekly ADX filter helps avoid whipsaws in ranging markets.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Donchian20_DailyEMA50_WeeklyADX_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data once for Donchian bands and EMA trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Get weekly data once for ADX trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    # Calculate daily Donchian bands (20-period)
    # Upper band = 20-period high, Lower band = 20-period low
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    donchian_high = pd.Series(daily_high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(daily_low).rolling(window=20, min_periods=20).min().values
    
    # Calculate daily EMA(50) for trend filter
    daily_close = df_1d['close'].values
    ema50_1d = pd.Series(daily_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate weekly ADX(14) for trend strength filter
    # ADX calculation: +DM, -DM, TR, then smoothed to get DX, then ADX
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Calculate True Range (TR)
    tr1 = weekly_high - weekly_low
    tr2 = np.abs(weekly_high - np.roll(weekly_close, 1))
    tr3 = np.abs(weekly_low - np.roll(weekly_close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period TR is just high-low
    
    # Calculate Directional Movement (+DM and -DM)
    up_move = weekly_high - np.roll(weekly_high, 1)
    down_move = np.roll(weekly_low, 1) - weekly_low
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smooth TR, +DM, -DM using Wilder's smoothing (similar to EMA with alpha=1/period)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            # First value is simple average
            result[period-1] = np.nanmean(data[:period])
            # Subsequent values: smoothed = prev * (1 - 1/period) + current * (1/period)
            for i in range(period, len(data)):
                if not np.isnan(result[i-1]):
                    result[i] = result[i-1] * (1 - 1/period) + data[i] * (1/period)
        return result
    
    atr = wilders_smoothing(tr, 14)
    plus_di = 100 * wilders_smoothing(plus_dm, 14) / atr
    minus_di = 100 * wilders_smoothing(minus_dm, 14) / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilders_smoothing(dx, 14)
    
    # Align all HTF indicators to 4h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # Volume spike: current volume > 1.5 * 20-period average on 4h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(adx_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        upper_band = donchian_high_aligned[i]
        lower_band = donchian_low_aligned[i]
        ema50_1d_val = ema50_1d_aligned[i]
        adx_val = adx_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: price breaks above upper band + daily uptrend + strong trend (ADX>25) + volume spike
            if (close[i] > upper_band and close[i] > ema50_1d_val and 
                adx_val > 25 and vol_spike):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below lower band + daily downtrend + strong trend (ADX>25) + volume spike
            elif (close[i] < lower_band and close[i] < ema50_1d_val and 
                  adx_val > 25 and vol_spike):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below lower band OR daily trend turns down
            if close[i] < lower_band or close[i] < ema50_1d_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above upper band OR daily trend turns up
            if close[i] > upper_band or close[i] > ema50_1d_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals