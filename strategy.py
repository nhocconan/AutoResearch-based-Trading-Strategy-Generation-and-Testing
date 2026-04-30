#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d Donchian(20) breakout with 1w EMA(50) trend filter and volume confirmation
# Donchian breakouts capture strong momentum moves. The 1w EMA(50) ensures trades align with the higher-timeframe trend,
# reducing false breakouts in choppy markets. Volume confirmation adds conviction to breakouts.
# Designed for low trade frequency (~12-37/year on 12h) to minimize fee drag and improve test generalization.

name = "12h_1dDonchian20_1wEMA50_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d Donchian channels (20-period high/low)
    high_1d = pd.Series(df_1d['high'].values)
    low_1d = pd.Series(df_1d['low'].values)
    donchian_high = high_1d.rolling(window=20, min_periods=20).max().values
    donchian_low = low_1d.rolling(window=20, min_periods=20).min().values
    
    # Align 1d Donchian channels to 12h timeframe (wait for 1d bar to close)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Calculate 1w EMA(50) for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    close_1w_s = pd.Series(df_1w['close'].values)
    ema_50_1w = close_1w_s.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate ATR(14) for dynamic stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 60  # warmup for EMA(50) and Donchian(20)
    
    for i in range(start_idx, n):
        # Volume confirmation: volume > 1.5x 20-period average
        vol_ma_20 = np.mean(volume[max(0, i-20):i]) if i >= 20 else np.mean(volume[:i]) if i > 0 else 0
        volume_spike = volume[i] > (1.5 * vol_ma_20) if i > 0 else False
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema = ema_50_1w_aligned[i]
        curr_atr = atr[i]
        curr_dch_high = donchian_high_aligned[i]
        curr_dch_low = donchian_low_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike and trend alignment
            if volume_spike:
                # Bullish entry: price breaks above 1d Donchian high with 1w uptrend
                if curr_close > curr_dch_high and curr_close > curr_ema:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish entry: price breaks below 1d Donchian low with 1w downtrend
                elif curr_close < curr_dch_low and curr_close < curr_ema:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Stoploss: 2.0 * ATR below entry price OR price breaks 1d Donchian low (reversal signal)
            if curr_close < entry_price - 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            elif curr_close < curr_dch_low:
                signals[i] = 0.0
                position = 0
            # Take profit: price reaches 1.5x ATR profit target
            elif curr_close >= entry_price + 3.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: 2.0 * ATR above entry price OR price breaks 1d Donchian high (reversal signal)
            if curr_close > entry_price + 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            elif curr_close > curr_dch_high:
                signals[i] = 0.0
                position = 0
            # Take profit: price reaches 1.5x ATR profit target
            elif curr_close <= entry_price - 3.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals