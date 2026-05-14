#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 12h Donchian(20) breakout with 12h HMA(21) trend filter and volume confirmation
# Uses 12h HTF for Donchian channels (price structure) and HMA trend filter to avoid whipsaws.
# Breakouts above upper channel in uptrend or below lower channel in downtrend with volume spike.
# Designed for low trade frequency (~19-50/year on 4h) to minimize fee drag while capturing strong directional moves.
# Works in bull markets via breakout continuation and in bear markets via mean-reversion at extreme levels.
# Focus on BTC/ETH as primary targets.

name = "4h_12hDonchian20_12hHMA21_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop for Donchian and HMA calculations
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 21:
        return np.zeros(n)
    
    # Calculate 12h Donchian channels (20-period)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    donchian_upper = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Align 12h Donchian channels to 4h timeframe (wait for 12h bar to close)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_12h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_12h, donchian_lower)
    
    # Calculate 12h HMA(21) for trend filter
    close_12h = df_12h['close'].values
    half_period = 21 // 2
    sqrt_period = int(np.sqrt(21))
    
    wma_half = pd.Series(close_12h).rolling(window=half_period, min_periods=half_period).apply(
        lambda x: np.average(x, weights=np.arange(1, len(x)+1)), raw=True
    ).values
    wma_full = pd.Series(close_12h).rolling(window=21, min_periods=21).apply(
        lambda x: np.average(x, weights=np.arange(1, len(x)+1)), raw=True
    ).values
    wma_sqrt = pd.Series(2 * wma_half - wma_full).rolling(window=sqrt_period, min_periods=sqrt_period).apply(
        lambda x: np.average(x, weights=np.arange(1, len(x)+1)), raw=True
    ).values
    hma_21_12h = wma_sqrt
    
    # Align 12h HMA to 4h timeframe
    hma_21_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_21_12h)
    
    # Calculate ATR(14) for dynamic stoploss on 4h
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 100  # warmup for Donchian(20) and HMA(21)
    
    for i in range(start_idx, n):
        # Volume confirmation: volume > 1.8x 30-period average (stricter to reduce trades)
        vol_ma_30 = np.mean(volume[max(0, i-30):i]) if i >= 30 else np.mean(volume[:i]) if i > 0 else 0
        volume_spike = volume[i] > (1.8 * vol_ma_30) if i > 0 else False
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_atr = atr[i]
        curr_upper = donchian_upper_aligned[i]
        curr_lower = donchian_lower_aligned[i]
        curr_hma = hma_21_12h_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike and trend alignment
            if volume_spike:
                # Bullish entry: price breaks above 12h upper Donchian with 12h uptrend (price > HMA)
                if curr_close > curr_upper and curr_close > curr_hma:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish entry: price breaks below 12h lower Donchian with 12h downtrend (price < HMA)
                elif curr_close < curr_lower and curr_close < curr_hma:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Stoploss: 2.5 * ATR below entry price OR price breaks 12h lower Donchian (reversal signal)
            if curr_close < entry_price - 2.5 * curr_atr:
                signals[i] = 0.0
                position = 0
            elif curr_close < curr_lower:
                signals[i] = 0.0
                position = 0
            # Take profit: price reaches 1.5x ATR above entry OR touches 12h upper Donchian (mean reversion)
            elif curr_close > entry_price + 1.5 * curr_atr:
                signals[i] = 0.10  # reduce position
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: 2.5 * ATR above entry price OR price breaks 12h upper Donchian (reversal signal)
            if curr_close > entry_price + 2.5 * curr_atr:
                signals[i] = 0.0
                position = 0
            elif curr_close > curr_upper:
                signals[i] = 0.0
                position = 0
            # Take profit: price reaches 1.5x ATR below entry OR touches 12h lower Donchian (mean reversion)
            elif curr_close < entry_price - 1.5 * curr_atr:
                signals[i] = -0.10  # reduce position
            else:
                signals[i] = -0.25
    
    return signals