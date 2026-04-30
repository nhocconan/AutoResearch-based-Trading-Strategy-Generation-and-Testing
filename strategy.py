#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Donchian channel breakout with 1d EMA50 trend filter and volume spike confirmation.
# Uses 4h HTF for Donchian channel (20-period) for structure and breakout signals, and 1d EMA50 for trend filter to avoid counter-trend trades.
# Long when price breaks above 4h upper Donchian channel in uptrend (1h close > 1d EMA50) with volume spike (>2.0x average).
# Short when price breaks below 4h lower Donchian channel in downtrend (1h close < 1d EMA50) with volume spike.
# Designed for moderate trade frequency (~15-37/year on 1h) to minimize fee drag while capturing strong directional moves.
# Uses volume confirmation (>2.0x average) and trend alignment to filter false breakouts.
# Stoploss via signal=0 when price reverses and breaks the opposite Donchian channel level.
# Works in bull markets via breakout continuation and in bear markets via fade of false breakouts at 4h channel levels.

name = "1h_4hDonchian20_Breakout_1dEMA50_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 4h and 1d data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 30 or len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 4h Donchian channel (20-period)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Donchian upper = max(high, lookback=20), lower = min(low, lookback=20)
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align 4h Donchian levels to 1h timeframe (wait for 4h bar to close)
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    # Calculate 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 50  # warmup for Donchian(20) and EMA(50)
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Volume confirmation: volume > 2.0x 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-20:i])
        elif i > 0:
            vol_ma_20 = np.mean(volume[:i])
        else:
            vol_ma_20 = 0
        volume_spike = volume[i] > (2.0 * vol_ma_20) if i > 0 else False
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_dch = donchian_high_aligned[i]
        curr_dcl = donchian_low_aligned[i]
        curr_ema = ema_50_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike and trend alignment
            if volume_spike:
                # Bullish entry: price breaks above 4h upper Donchian with 1d uptrend (close > EMA50)
                if curr_close > curr_dch and curr_close > curr_ema:
                    signals[i] = 0.20
                    position = 1
                    entry_price = curr_close
                # Bearish entry: price breaks below 4h lower Donchian with 1d downtrend (close < EMA50)
                elif curr_close < curr_dcl and curr_close < curr_ema:
                    signals[i] = -0.20
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Exit: price breaks below 4h lower Donchian channel (reversal signal)
            if curr_close < curr_dcl:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit: price breaks above 4h upper Donchian channel (reversal signal)
            if curr_close > curr_dch:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals