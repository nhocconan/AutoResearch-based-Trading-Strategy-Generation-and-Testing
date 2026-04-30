#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 12h Donchian channel breakout with 1d EMA50 trend filter and volume spike confirmation
# Uses 12h HTF for Donchian channel (structure) to avoid 6h noise and 1d EMA50 for trend to filter false breakouts.
# Long when price breaks above 12h Donchian upper (20) in uptrend (1d close > 1d EMA50) with volume spike (>2.0x average).
# Short when price breaks below 12h Donchian lower (20) in downtrend (1d close < 1d EMA50) with volume spike.
# Designed for low trade frequency (~15-35/year on 6h) to minimize fee drag while capturing strong directional moves.
# Uses volume confirmation with moderate threshold (>2.0x average) to balance signal quality and trade count.
# Stoploss via signal=0 when price reverses and breaks opposite Donchian level (structure-based).
# Works in bull markets via breakout continuation and in bear markets via fade of false breakouts at 12h structure.
# Focus on BTC/ETH as primary targets.

name = "6h_12hDonchian20_Breakout_1dEMA50_VolumeSpike_v1"
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
    
    # Load 12h data ONCE before loop for Donchian calculations
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate 12h Donchian channel (20-period)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Donchian upper = max(high, lookback=20), lower = min(low, lookback=20)
    donchian_upper = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Align 12h Donchian levels to 6h timeframe (wait for 12h bar to close)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_12h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_12h, donchian_lower)
    
    # Calculate 1d EMA(50) for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate ATR(14) for 6h timeframe (used for volume spike averaging)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for EMA(50) and Donchian(20)
    
    for i in range(start_idx, n):
        # Volume confirmation: volume > 2.0x 50-period average (moderate to balance trades)
        if i >= 50:
            vol_ma_50 = np.mean(volume[i-50:i])
        elif i > 0:
            vol_ma_50 = np.mean(volume[:i])
        else:
            vol_ma_50 = 0
        volume_spike = volume[i] > (2.0 * vol_ma_50) if i > 0 else False
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_vol_spike = volume_spike
        curr_upper = donchian_upper_aligned[i]
        curr_lower = donchian_lower_aligned[i]
        curr_ema = ema_50_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike and trend alignment
            if curr_vol_spike:
                # Bullish entry: price breaks above 12h Donchian upper with 1d uptrend (close > EMA50)
                if curr_close > curr_upper and curr_close > curr_ema:
                    signals[i] = 0.25
                    position = 1
                # Bearish entry: price breaks below 12h Donchian lower with 1d downtrend (close < EMA50)
                elif curr_close < curr_lower and curr_close < curr_ema:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position
            # Exit: price breaks below 12h Donchian lower (structure break) OR reverse signal
            if curr_close < curr_lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price breaks above 12h Donchian upper (structure break) OR reverse signal
            if curr_close > curr_upper:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals