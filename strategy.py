#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1d weekly pivot confirmation and volume filter.
# Long when price breaks above 6h Donchian upper channel AND price > 1d weekly pivot R1 AND volume > 1.5x 20-period 6h average.
# Short when price breaks below 6h Donchian lower channel AND price < 1d weekly pivot S1 AND volume > 1.5x 20-period 6h average.
# Exit on Donchian middle line retrace or ATR(14) stoploss (2*ATR).
# Uses discrete position size 0.25. Designed to capture breakouts in trending markets with pivot confirmation.
# Weekly pivot from 1d data provides institutional reference points that work in both bull and bear markets.
# Volume filter ensures breakouts have participation, reducing false signals.
# Target: 80-180 total trades over 4 years (20-45/year) to balance edge and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 6h Indicators: Donchian Channel (20-period) ===
    donchian_window = 20
    donchian_high = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    donchian_middle = (donchian_high + donchian_low) / 2
    
    # === 1d Indicators: Weekly Pivot Points (using prior week's OHLC) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly OHLC from daily data
    # Group by week (starting Monday) and get weekly OHLC
    df_1d_temp = pd.DataFrame({
        'high': high_1d,
        'low': low_1d,
        'close': close_1d
    }, index=pd.to_datetime(df_1d.index))  # df_1d index is already DatetimeIndex from get_htf_data
    
    # Resample to weekly (Monday start)
    weekly = df_1d_temp.resample('W-MON').agg({
        'high': 'max',
        'low': 'min',
        'close': 'last'
    })
    weekly_open = df_1d_temp.resample('W-MON').first()['open'] if 'open' in df_1d_temp.columns else weekly['close']  # approximate open as prior week close
    
    # Since we don't have open in 1d data from get_htf_data, use close as approximation for weekly open
    weekly_open = weekly['close'].shift(1)  # prior week's close as this week's open
    weekly_high = weekly['high']
    weekly_low = weekly['low']
    weekly_close = weekly['close']
    
    # Calculate weekly pivot points
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3
    weekly_r1 = 2 * weekly_pivot - weekly_low
    weekly_s1 = 2 * weekly_pivot - weekly_high
    weekly_r2 = weekly_pivot + (weekly_high - weekly_low)
    weekly_s2 = weekly_pivot - (weekly_high - weekly_low)
    weekly_r3 = weekly_high + 2 * (weekly_pivot - weekly_low)
    weekly_s3 = weekly_low - 2 * (weekly_high - weekly_pivot)
    
    # Align weekly pivot levels to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot.values)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1d, weekly_r1.values)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1d, weekly_s1.values)
    
    # === 6h Indicators: Volume Spike (volume > 1.5x 20-period average) ===
    vol_ma_6h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_6h)
    
    # === 6h ATR for stoploss ===
    tr1_6h = pd.Series(high).diff()
    tr2_6h = pd.Series(low).diff().abs()
    tr3_6h = pd.Series(close).shift(1).diff().abs()
    tr_6h = pd.concat([tr1_6h, tr2_6h, tr3_6h], axis=1).max(axis=1)
    atr_6h_raw = pd.Series(tr_6h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    # Track position state and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(donchian_middle[i]) or
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(weekly_r1_aligned[i]) or np.isnan(weekly_s1_aligned[i]) or
            np.isnan(volume_spike[i]) or np.isnan(atr_6h_raw[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        vol_spike = volume_spike[i]
        atr_val = atr_6h_raw[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price retraces to Donchian middle line
            if price <= donchian_middle[i]:
                exit_signal = True
            # ATR-based stoploss: 2*ATR below entry
            elif price < entry_price - 2.0 * atr_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price retraces to Donchian middle line
            if price >= donchian_middle[i]:
                exit_signal = True
            # ATR-based stoploss: 2*ATR above entry
            elif price > entry_price + 2.0 * atr_val:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above Donchian upper channel AND price > weekly R1 AND volume spike
            if close[i] > donchian_high[i] and close[i] > weekly_r1_aligned[i] and vol_spike:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: Price breaks below Donchian lower channel AND price < weekly S1 AND volume spike
            elif close[i] < donchian_low[i] and close[i] < weekly_s1_aligned[i] and vol_spike:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "6h_Donchian20_1dWeeklyPivot_VolumeSpike_V1"
timeframe = "6h"
leverage = 1.0