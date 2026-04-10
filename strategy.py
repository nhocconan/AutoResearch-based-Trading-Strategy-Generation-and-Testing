#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian channel breakout with 12h ADX trend filter and 1d volume spike filter
# - Entry: Long when price breaks above 4h Donchian upper channel (20) + 1d volume > 2.0x 20-period average + 12h ADX(14) > 25
#          Short when price breaks below 4h Donchian lower channel (20) + same volume and trend filters
# - Exit: Close-based reversal - exit long when price < 4h Donchian lower channel, exit short when price > 4h Donchian upper channel
# - Stoploss: ATR-based - exit when price moves against position by 2.5 * ATR(14) on 4h
# - Position sizing: 0.25 (discrete level)
# - Volume spike filter (2.0x average) is stricter than typical 1.5x to reduce false breakouts and trade frequency
# - Higher ATR multiplier (2.5) reduces whipsaw exits in choppy markets
# - Target: 75-150 total trades over 4 years (19-38/year) to stay well within HARD MAX: 400 total

name = "4h_1d_12h_donchian_breakout_volume_adx_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_12h = get_htf_data(prices, '12h')
    if len(df_1d) < 30 or len(df_12h) < 30:
        return np.zeros(n)
    
    # Pre-compute 4h OHLC
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    
    # Pre-compute 1d data for volume
    volume_1d = df_1d['volume'].values
    
    # Pre-compute 12h data for ADX
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 4h Donchian channels (20-period)
    donchian_high_4h = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low_4h = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d volume moving average (20-period)
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ma_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    
    # Calculate 12h ADX (14-period) for trend filter
    # Calculate directional movement
    high_diff = high_12h - np.roll(high_12h, 1)
    low_diff = np.roll(low_12h, 1) - low_12h
    
    plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0.0)
    minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0.0)
    
    # Set first values to 0
    plus_dm[0] = 0.0
    minus_dm[0] = 0.0
    
    # Calculate True Range
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = 0
    
    # Wilder's smoothing function
    def wilders_smoothing(data, period):
        alpha = 1.0 / period
        result = np.zeros_like(data)
        if len(data) >= period:
            result[period-1] = np.mean(data[:period])  # First value is simple average
            for i in range(period, len(data)):
                result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        return result
    
    # Smooth TR, +DM, -DM
    atr_12h = wilders_smoothing(tr, 14)
    plus_di_12h = 100 * wilders_smoothing(plus_dm, 14) / atr_12h
    minus_di_12h = 100 * wilders_smoothing(minus_dm, 14) / atr_12h
    
    # Handle division by zero
    plus_di_12h = np.where(atr_12h == 0, 0, plus_di_12h)
    minus_di_12h = np.where(atr_12h == 0, 0, minus_di_12h)
    
    # Calculate DX and ADX
    dx_12h = 100 * np.abs(plus_di_12h - minus_di_12h) / (plus_di_12h + minus_di_12h)
    dx_12h = np.where((plus_di_12h + minus_di_12h) == 0, 0, dx_12h)  # Handle division by zero
    adx_12h = wilders_smoothing(dx_12h, 14)
    
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
            np.isnan(volume_ma_aligned[i]) or np.isnan(adx_12h[i]) or 
            np.isnan(atr_14_4h[i])):
            signals[i] = 0.0
            continue
        
        # Get current 4h close
        close_price = close_4h[i]
        
        # Get current 1d volume for confirmation
        volume_1d_current = align_htf_to_ltf(prices, df_1d, volume_1d)[i]
        volume_confirmation = volume_1d_current > 2.0 * volume_ma_aligned[i]
        
        # Trend filter: ADX > 25 indicates trending market (avoid choppy conditions)
        trend_filter = adx_12h[i] > 25
        
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
                stop_loss = entry_price - 2.5 * atr_14_4h[i]
                # Exit conditions: price < Donchian lower channel OR stoploss hit
                if close_price < donchian_low_4h[i] or close_price <= stop_loss:
                    position = 0
                    entry_price = 0.0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1, Short position
                stop_loss = entry_price + 2.5 * atr_14_4h[i]
                # Exit conditions: price > Donchian upper channel OR stoploss hit
                if close_price > donchian_high_4h[i] or close_price >= stop_loss:
                    position = 0
                    entry_price = 0.0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals