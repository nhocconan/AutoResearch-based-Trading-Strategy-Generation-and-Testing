#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Donchian breakout (20) with 1d EMA50 trend filter and volume spike confirmation
# Uses 4h HTF for Donchian channel calculation (upper/lower 20-period) for structure and 1d EMA50 for trend filter to avoid counter-trend trades.
# Long when price breaks above 4h Donchian upper in uptrend (1h close > 1d EMA50) with volume spike (>2.0x average).
# Short when price breaks below 4h Donchian lower in downtrend (1h close < 1d EMA50) with volume spike.
# Designed for moderate trade frequency (~30-60/year on 1h) to minimize fee drag while capturing strong directional moves.
# Uses volume confirmation with moderate threshold (>2.0x average) to balance signal quality and trade count.
# Stoploss via signal=0 when price reverses and closes back inside the 4h Donchian channel.
# Works in bull markets via breakout continuation and in bear markets via fade of false breakouts at 4h Donchian levels.
# Focus on BTC/ETH as primary targets.

name = "1h_4hDonchian20_Breakout_1dEMA50_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 4h data ONCE before loop for Donchian calculations
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Calculate 4h Donchian channel (20-period)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align 4h Donchian levels to 1h timeframe (wait for 4h bar to close)
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    # Load 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Volume spike: >2.0x 50-period average (pre-computed for efficiency)
    volume_ma_50 = np.zeros(n)
    for i in range(50, n):
        volume_ma_50[i] = np.mean(volume[i-50:i])
    for i in range(1, 50):
        volume_ma_50[i] = np.mean(volume[:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 60)  # warmup for EMA(50) and volume MA
    
    for i in range(start_idx, n):
        # Session filter: 08-20 UTC only
        hour = pd.Timestamp(prices['open_time'].iloc[i]).hour
        if hour < 8 or hour > 20:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: volume > 2.0x 50-period average
        volume_spike = volume[i] > (2.0 * volume_ma_50[i])
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_dc_high = donchian_high_aligned[i]
        curr_dc_low = donchian_low_aligned[i]
        curr_ema = ema_50_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike and trend alignment
            if volume_spike:
                # Bullish entry: price breaks above 4h Donchian upper with 1d uptrend (close > EMA50)
                if curr_close > curr_dc_high and curr_close > curr_ema:
                    signals[i] = 0.20
                    position = 1
                # Bearish entry: price breaks below 4h Donchian lower with 1d downtrend (close < EMA50)
                elif curr_close < curr_dc_low and curr_close < curr_ema:
                    signals[i] = -0.20
                    position = -1
        
        elif position == 1:  # Long position
            # Exit: price closes back inside 4h Donchian channel (reversal signal)
            if curr_close < curr_dc_high and curr_close > curr_dc_low:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit: price closes back inside 4h Donchian channel (reversal signal)
            if curr_close < curr_dc_high and curr_close > curr_dc_low:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals