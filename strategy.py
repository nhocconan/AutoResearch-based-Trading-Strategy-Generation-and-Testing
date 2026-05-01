#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume spike and ADX regime filter.
# Long when price breaks above Donchian(20) upper band with volume > 2.0x 4h volume average and ADX(14) > 25.
# Short when price breaks below Donchian(20) lower band with volume confirmation and ADX > 25.
# Uses discrete sizing 0.25. ATR(14) stoploss: signal→0 when price moves against position by 2.0*ATR.
# Donchian bands calculated from prior completed 4h bar (no look-ahead). Target: 20-50 trades/year on 4h timeframe.
# Volume spike ensures momentum behind breakout. ADX ensures trades only in trending regimes, reducing whipsaw in chop.
# Works in bull (strong uptrend breakouts) and bear (strong downtrend breakouts) markets.

name = "4h_Donchian20_Breakout_Volume_ADX_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr_first = np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])
    tr = np.concatenate([[tr_first], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Load 4h data ONCE before loop for volume filter and Donchian bands
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4h volume average
    vol_4h = df_4h['volume'].values
    vol_ma_4h = pd.Series(vol_4h).rolling(window=20, min_periods=20).mean().values
    
    # Align 4h volume MA to 4h timeframe (no shift needed as already aligned)
    vol_ma_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_4h)
    
    # Calculate Donchian(20) bands on 4h data (using previous completed bar)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Upper band: max high of previous 20 completed 4h bars
    upper_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    # Lower band: min low of previous 20 completed 4h bars
    lower_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Shift to align with current bar (use previous 20 bars, not including current)
    upper_20 = np.concatenate([[np.nan], upper_20[:-1]])
    lower_20 = np.concatenate([[np.nan], lower_20[:-1]])
    
    # Load 1h data ONCE before loop for ADX trend filter (HTF)
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 14:
        return np.zeros(n)
    
    # Calculate ADX(14) on 1h data
    high_1h = df_1h['high'].values
    low_1h = df_1h['low'].values
    close_1h = df_1h['close'].values
    
    # True Range
    tr1_1h = high_1h[1:] - low_1h[1:]
    tr2_1h = np.abs(high_1h[1:] - close_1h[:-1])
    tr3_1h = np.abs(low_1h[1:] - close_1h[:-1])
    tr_first_1h = np.max([high_1h[0] - low_1h[0], np.abs(high_1h[0] - close_1h[0]), np.abs(low_1h[0] - close_1h[0])])
    tr_1h = np.concatenate([[tr_first_1h], np.maximum(tr1_1h, np.maximum(tr2_1h, tr3_1h))])
    
    # Directional Movement
    dm_plus_1h = np.where((high_1h[1:] - high_1h[:-1]) > (low_1h[:-1] - low_1h[1:]), 
                          np.maximum(high_1h[1:] - high_1h[:-1], 0), 0)
    dm_minus_1h = np.where((low_1h[:-1] - low_1h[1:]) > (high_1h[1:] - high_1h[:-1]), 
                           np.maximum(low_1h[:-1] - low_1h[1:], 0), 0)
    dm_plus_1h = np.concatenate([[0], dm_plus_1h])
    dm_minus_1h = np.concatenate([[0], dm_minus_1h])
    
    # Smoothed TR, DM+, DM-
    tr_14 = pd.Series(tr_1h).ewm(alpha=1/14, adjust=False).mean().values
    dm_plus_14 = pd.Series(dm_plus_1h).ewm(alpha=1/14, adjust=False).mean().values
    dm_minus_14 = pd.Series(dm_minus_1h).ewm(alpha=1/14, adjust=False).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    
    # Align ADX to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1h, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # track entry price for stoploss
    
    # Start after warmup for ATR, Donchian, volume, and ADX
    start_idx = 40
    
    for i in range(start_idx, n):
        if (np.isnan(atr[i]) or 
            np.isnan(vol_ma_4h_aligned[i]) or 
            np.isnan(upper_20[i]) or 
            np.isnan(lower_20[i]) or 
            np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Volume confirmation: current volume > 2.0x 4h volume average
        if vol_ma_4h_aligned[i] <= 0 or np.isnan(vol_ma_4h_aligned[i]):
            volume_confirm = False
        else:
            volume_confirm = curr_volume > (vol_ma_4h_aligned[i] * 2.0)
        
        # Trend filter: ADX > 25 (trending market)
        trend_filter = adx_aligned[i] > 25
        
        upper_level = upper_20[i]
        lower_level = lower_20[i]
        
        # Donchian breakout conditions
        breakout_up = curr_high > upper_level  # break above upper band
        breakout_down = curr_low < lower_level  # break below lower band
        
        if position == 0:  # Flat - look for new entries
            # Long: Donchian breakout up AND volume confirmation AND trend filter
            if (breakout_up and 
                volume_confirm and 
                trend_filter):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: Donchian breakout down AND volume confirmation AND trend filter
            elif (breakout_down and 
                  volume_confirm and 
                  trend_filter):
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Stoploss: price moves against position by 2.0*ATR
            if curr_close < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price re-enters Donchian channel OR ADX < 20 (trend weakens)
            elif (curr_low >= lower_level and curr_low <= upper_level) or \
                 (adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: price moves against position by 2.0*ATR
            if curr_close > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price re-enters Donchian channel OR ADX < 20 (trend weakens)
            elif (curr_high >= lower_level and curr_high <= upper_level) or \
                 (adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
    
    return signals