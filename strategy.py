#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian breakout with 1w ADX trend filter and volume confirmation
# Long: Close > Donchian Upper(20) AND 1w ADX > 25 AND volume > 2.0x 20-bar avg
# Short: Close < Donchian Lower(20) AND 1w ADX > 25 AND volume > 2.0x 20-bar avg
# Exit: Close crosses Donchian midpoint OR 1w ADX < 20 (trend weakening)
# ATR stoploss: 2.0 * ATR(14) from entry price
# Works in bull via breakout continuation, in bear via strong trend capture
# Target: 50-150 total trades over 4 years (12-37/year) on 6h timeframe
# Discrete position sizing: 0.25 for long/short, 0.0 for flat to minimize fee churn

name = "6h_Donchian_Breakout_1wADX_VolumeSpike_ATRStop_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    # Calculate 1w ADX for trend filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w[1:] - low_1w[1:]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr_first = np.max([high_1w[0] - low_1w[0], np.abs(high_1w[0] - close_1w[0]), np.abs(low_1w[0] - close_1w[0])])
    tr_1w = np.concatenate([[tr_first], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_1w[1:] - high_1w[:-1]) > (low_1w[:-1] - low_1w[1:]), 
                       np.maximum(high_1w[1:] - high_1w[:-1], 0), 0)
    dm_minus = np.where((low_1w[:-1] - low_1w[1:]) > (high_1w[1:] - high_1w[:-1]), 
                        np.maximum(low_1w[:-1] - low_1w[1:], 0), 0)
    dm_plus_first = np.maximum(high_1w[0] - high_1w[0], 0)  # Always 0 for first bar
    dm_minus_first = np.maximum(low_1w[0] - low_1w[0], 0)   # Always 0 for first bar
    dm_plus = np.concatenate([[dm_plus_first], dm_plus])
    dm_minus = np.concatenate([[dm_minus_first], dm_minus])
    
    # Smoothed TR, DM+, DM- using Wilder's smoothing (alpha = 1/period)
    def wilder_smoothing(data, period):
        result = np.full_like(data, np.nan)
        alpha = 1.0 / period
        # First value is simple average
        result[period-1] = np.mean(data[:period])
        # Rest is Wilder smoothing
        for i in range(period, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        return result
    
    atr_1w = wilder_smoothing(tr_1w, 14)
    dm_plus_smooth = wilder_smoothing(dm_plus, 14)
    dm_minus_smooth = wilder_smoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / (atr_1w + 1e-10)
    di_minus = 100 * dm_minus_smooth / (atr_1w + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = wilder_smoothing(dx, 14)
    
    # Align to LTF
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # Calculate ATR for stoploss (using 14-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr_first = np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])
    tr = np.concatenate([[tr_first], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(34, 20, 14)  # warmup for indicators
    
    for i in range(start_idx, n):
        # Calculate Donchian channels (20-period)
        if i >= 20:
            donch_high = np.max(high[i-20:i])
            donch_low = np.min(low[i-20:i])
            donch_mid = (donch_high + donch_low) / 2.0
        else:
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_adx_1w = adx_1w_aligned[i]
        curr_atr = atr[i]
        
        # Volume spike confirmation: current volume > 2.0x 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-20:i])
        else:
            vol_ma_20 = 0.0
        vol_spike = volume[i] > 2.0 * vol_ma_20 if vol_ma_20 > 0 else False
        
        # Handle exits and stoploss
        if position == 1:  # Long position
            # Stoploss: 2 * ATR below entry
            stop_price = entry_price - 2.0 * curr_atr
            # Exit conditions: Close below Donchian mid OR ADX < 20 (trend weakening) OR stoploss hit
            if curr_close < donch_mid or curr_adx_1w < 20.0 or curr_close < stop_price:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Stoploss: 2 * ATR above entry
            stop_price = entry_price + 2.0 * curr_atr
            # Exit conditions: Close above Donchian mid OR ADX < 20 (trend weakening) OR stoploss hit
            if curr_close > donch_mid or curr_adx_1w < 20.0 or curr_close > stop_price:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Only enter when ADX > 25 (strong trend)
            if curr_adx_1w > 25.0:
                # Long entry: Close > Donchian Upper AND volume spike
                if curr_close > donch_high and vol_spike:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Short entry: Close < Donchian Lower AND volume spike
                elif curr_close < donch_low and vol_spike:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0
    
    return signals