#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian channel breakout with 12h volume confirmation and 1d ADX trend filter
# - Entry: Long when price breaks above 4h Donchian upper channel (20) + 12h volume > 1.5x 20-period average + 1d ADX(14) > 25
#          Short when price breaks below 4h Donchian lower channel (20) + same volume and trend filters
# - Exit: Close-based reversal - exit long when price < 4h Donchian lower channel, exit short when price > 4h Donchian upper channel
# - Stoploss: ATR-based - exit when price moves against position by 2.0 * ATR(14) on 4h
# - Position sizing: 0.25 (discrete level)
# - Donchian channels provide clear breakout levels that work in both trending and ranging markets
# - Volume confirmation on 12h avoids false breakouts, ADX filter on 1d ensures we trade in trending regimes
# - Target: 100-200 total trades over 4 years (25-50/year) to stay within HARD MAX: 400 total

name = "4h_12h_1d_donchian_breakout_volume_adx_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_12h) < 30 or len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 4h OHLC
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    
    # Pre-compute 12h data for volume
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Pre-compute 1d data for ADX
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 4h Donchian channels (20-period)
    # Upper channel = highest high over last 20 periods
    # Lower channel = lowest low over last 20 periods
    donchian_high_4h = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low_4h = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 4h (already aligned as we calculated on 4h data)
    # No need to align as we're using 4h data directly
    
    # Calculate 12h volume moving average (20-period)
    volume_ma_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    volume_ma_aligned = align_htf_to_ltf(prices, df_12h, volume_ma_20_12h)
    
    # Calculate 1d ADX (14-period) for trend filter
    # +DM = max(high - prev_high, 0) if high - prev_high > prev_low - low else 0
    # -DM = max(prev_low - low, 0) if prev_low - low > high - prev_high else 0
    # TR = max(high - low, abs(high - prev_close), abs(low - prev_close))
    # +DI = 100 * EWMA(+DM) / EWMA(TR)
    # -DI = 100 * EWMA(-DM) / EWMA(TR)
    # DX = 100 * abs(+DI - -DI) / (+DI + -DI)
    # ADX = EWMA(DX)
    
    # Calculate directional movement
    high_diff = high_1d - np.roll(high_1d, 1)
    low_diff = np.roll(low_1d, 1) - low_1d
    
    plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0.0)
    minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0.0)
    
    # Set first values to 0
    plus_dm[0] = 0.0
    minus_dm[0] = 0.0
    
    # Calculate True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = 0
    
    # Calculate smoothed values using Wilder's smoothing (EMA with alpha=1/period)
    def wilders_smoothing(data, period):
        alpha = 1.0 / period
        result = np.zeros_like(data)
        if len(data) >= period:
            result[period-1] = np.mean(data[:period])  # First value is simple average
            for i in range(period, len(data)):
                result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        return result
    
    # Smooth TR, +DM, -DM
    atr_1d = wilders_smoothing(tr, 14)
    plus_di_1d = 100 * wilders_smoothing(plus_dm, 14) / atr_1d
    minus_di_1d = 100 * wilders_smoothing(minus_dm, 14) / atr_1d
    
    # Handle division by zero
    plus_di_1d = np.where(atr_1d == 0, 0, plus_di_1d)
    minus_di_1d = np.where(atr_1d == 0, 0, minus_di_1d)
    
    # Calculate DX and ADX
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)
    dx_1d = np.where((plus_di_1d + minus_di_1d) == 0, 0, dx_1d)  # Handle division by zero
    adx_1d = wilders_smoothing(dx_1d, 14)
    
    # Calculate 4h ATR (14-period) for stoploss
    tr1_4h = high_4h - low_4h
    tr2_4h = np.abs(high_4h - np.roll(close_4h, 1))
    tr3_4h = np.abs(low_4h - np.roll(close_4h, 1))
    tr_4h = np.maximum(tr1_4h, np.maximum(tr2_4h, tr3_4h))
    tr_4h[0] = 0
    atr_14_4h = wilders_smoothing(tr_4h, 14)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(50, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(donchian_high_4h[i]) or np.isnan(donchian_low_4h[i]) or 
            np.isnan(volume_ma_aligned[i]) or np.isnan(adx_1d[i]) or 
            np.isnan(atr_14_4h[i])):
            signals[i] = 0.0
            continue
        
        # Get current 4h close
        close_price = close_4h[i]
        
        # Get current 12h volume for confirmation
        volume_12h_current = align_htf_to_ltf(prices, df_12h, volume_12h)[i]
        volume_confirmation = volume_12h_current > 1.5 * volume_ma_aligned[i]
        
        # Trend filter: ADX > 25 indicates trending market (avoid choppy conditions)
        trend_filter = adx_1d[i] > 25
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above Donchian upper channel + volume confirmation + trend filter
            if (close_price > donchian_high_4h[i] and 
                volume_confirmation and 
                trend_filter):
                position = 1
                entry_price = close_price
                signals[i] = 0.25
            # Short entry: price breaks below Donchian lower channel + volume confirmation + trend filter
            elif (close_price < donchian_low_4h[i] and 
                  volume_confirmation and 
                  trend_filter):
                position = -1
                entry_price = close_price
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit or stoploss
            # Calculate stoploss level
            if position == 1:  # Long position
                stop_loss = entry_price - 2.0 * atr_14_4h[i]
                # Exit conditions: price < Donchian lower channel OR stoploss hit
                if close_price < donchian_low_4h[i] or close_price <= stop_loss:
                    position = 0
                    entry_price = 0.0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1, Short position
                stop_loss = entry_price + 2.0 * atr_14_4h[i]
                # Exit conditions: price > Donchian upper channel OR stoploss hit
                if close_price > donchian_high_4h[i] or close_price >= stop_loss:
                    position = 0
                    entry_price = 0.0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals