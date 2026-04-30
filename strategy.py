#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d EMA50 trend filter and volume confirmation
# Williams Alligator (Jaw=13, Teeth=8, Lips=5) identifies trend strength and direction.
# Only trade when all three lines are aligned (bullish or bearish) + price outside mouth.
# 1d EMA50 filters counter-trend moves. Volume spike ensures institutional participation.
# Designed for 12h timeframe to capture medium-term swings in both bull and bear markets.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.

name = "12h_WilliamsAlligator_1dEMA50_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Calculate Williams Alligator (SMMA = Smoothed Moving Average)
    def smma(data, period):
        result = np.full_like(data, np.nan, dtype=np.float64)
        if len(data) < period:
            return result
        # First value is simple SMA
        result[period-1] = np.mean(data[:period])
        # Subsequent values: SMMA = (prev SMMA * (period-1) + current) / period
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    # Alligator lines: Jaw(13), Teeth(8), Lips(5)
    jaw = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)
    
    # Calculate 1d EMA(50) for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: volume > 2.0x 24-period average (strict to reduce trades)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (2.0 * vol_ma_24)
    
    # ATR for stoploss (14-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(100, 24, 50, 14)  # warmup
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_24[i]) or
            np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_jaw = jaw[i]
        curr_teeth = teeth[i]
        curr_lips = lips[i]
        curr_ema_50_1d = ema_50_1d_aligned[i]
        curr_volume_spike = volume_spike[i]
        curr_atr = atr_14[i]
        
        # Alligator alignment: Bullish when Lips > Teeth > Jaw, Bearish when Lips < Teeth < Jaw
        bullish_aligned = curr_lips > curr_teeth > curr_jaw
        bearish_aligned = curr_lips < curr_teeth < curr_jaw
        
        if position == 0:  # Flat - look for new entries
            # Only trade on volume spike with Alligator alignment and 1d EMA50 trend filter
            if curr_volume_spike:
                # Bullish: Price above Alligator mouth (above Lips) + above 1d EMA50
                if bullish_aligned and curr_close > curr_lips and curr_close > curr_ema_50_1d:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish: Price below Alligator mouth (below Lips) + below 1d EMA50
                elif bearish_aligned and curr_close < curr_lips and curr_close < curr_ema_50_1d:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Stoploss: 2.5 * ATR below entry
            stop_loss = entry_price - 2.5 * curr_atr
            # Exit: Stoploss hit OR price drops below Teeth OR loses 1d trend
            if curr_low <= stop_loss or curr_close < curr_teeth or curr_close < curr_ema_50_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: 2.5 * ATR above entry
            stop_loss = entry_price + 2.5 * curr_atr
            # Exit: Stoploss hit OR price rises above Teeth OR loses 1d trend
            if curr_high >= stop_loss or curr_close > curr_teeth or curr_close > curr_ema_50_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals