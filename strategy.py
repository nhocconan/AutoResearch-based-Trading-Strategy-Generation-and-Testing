#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation
# Uses Donchian channel breakouts (20-period high/low) for trend continuation entries
# Only takes trades in direction of 1d EMA34 trend with volume spike (>1.8x average)
# ATR-based stoploss (2.0x ATR) and Donchian opposite channel exit
# Designed to work in both bull and bear markets by following major trend
# Target: 20-50 trades/year via tight Donchian breakout conditions + volume + trend filter

name = "4h_Donchian20_1dEMA34_Trend_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate EMA34 on 1d close for trend filter
    close_1d = pd.Series(df_1d['close'])
    ema34_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA34 to 4h timeframe (completed 1d candles only)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Get 4h data for Donchian channel calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Calculate Donchian(20) on 4h high/low
    high_4h = pd.Series(df_4h['high'])
    low_4h = pd.Series(df_4h['low'])
    donchian_high = high_4h.rolling(window=20, min_periods=20).max().values
    donchian_low = low_4h.rolling(window=20, min_periods=20).min().values
    
    # Align 4h Donchian channels to 4h timeframe (completed 4h candles only)
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    # Volume confirmation: >1.8x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 1.8 * volume_ma_20
    
    # Calculate ATR(14) for stoploss on 4h data
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - close_4h) if 'close_4h' in locals() else np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - close_4h) if 'close_4h' in locals() else np.abs(low_4h - np.roll(close_4h, 1))
    # Fix: need close_4h
    close_4h = df_4h['close'].values
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_aligned = align_htf_to_ltf(prices, df_4h, atr_14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(20, 34)  # Need sufficient history for Donchian and EMA34
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(ema34_1d_aligned[i]) or np.isnan(volume_ma_20[i]) or
            np.isnan(atr_14_aligned[i])):
            signals[i] = 0.0
            continue
        
        vol_confirm = volume_spike[i]
        price = close[i]
        upper = donchian_high_aligned[i]
        lower = donchian_low_aligned[i]
        ema34_val = ema34_1d_aligned[i]
        atr_val = atr_14_aligned[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long breakout: price breaks above upper Donchian AND 1d EMA34 uptrend AND volume spike
            if price > upper and price > ema34_val and vol_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short breakout: price breaks below lower Donchian AND 1d EMA34 downtrend AND volume spike
            elif price < lower and price < ema34_val and vol_confirm:
                signals[i] = -0.25
                position = -1
                entry_price = price
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit on stoploss or price falls below lower Donchian
            # ATR-based stoploss: 2.0 * ATR below entry
            stop_loss = entry_price - 2.0 * atr_val
            # Exit on stoploss or price < lower Donchian (trend reversal)
            if price < stop_loss or price < lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit on stoploss or price rises above upper Donchian
            # ATR-based stoploss: 2.0 * ATR above entry
            stop_loss = entry_price + 2.0 * atr_val
            # Exit on stoploss or price > upper Donchian (trend reversal)
            if price > stop_loss or price > upper:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals