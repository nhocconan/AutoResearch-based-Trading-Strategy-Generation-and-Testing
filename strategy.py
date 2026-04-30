#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Donchian(20) breakout with 1d HMA(21) trend filter and volume spike confirmation
# Uses 1d HTF for Donchian channel calculation (key support/resistance) and HMA trend to reduce whipsaws.
# Long when price breaks above 1d Donchian upper channel in uptrend (close > HMA21) with volume spike.
# Short when price breaks below 1d Donchian lower channel in downtrend (close < HMA21) with volume spike.
# Designed for low trade frequency (~20-35/year on 4h) to minimize fee drag while capturing strong directional moves.
# Works in bull markets via breakout continuation and in bear markets via fade of false breakouts at extreme levels.
# Focus on BTC/ETH as primary targets.

name = "4h_1dDonchian20_1dHMA21_Trend_VolumeSpike_v1"
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
    
    # Load 1d data ONCE before loop for Donchian and HMA calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 21:
        return np.zeros(n)
    
    # Calculate 1d Donchian channels (20-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Donchian Upper = max(high, 20), Donchian Lower = min(low, 20)
    donchian_upper = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align 1d Donchian channels to 4h timeframe (wait for 1d bar to close)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower)
    
    # Calculate 1d HMA(21) for trend filter
    # HMA = WMA(2 * WMA(n/2) - WMA(n)), sqrt(n)
    def calculate_wma(values, window):
        weights = np.arange(1, window + 1)
        return np.convolve(values, weights, 'valid') / weights.sum()
    
    def calculate_hma(values, window):
        half_window = window // 2
        sqrt_window = int(np.sqrt(window))
        
        wma_half = calculate_wma(values, half_window)
        wma_full = calculate_wma(values, window)
        
        raw_hma = 2 * wma_half - wma_full
        hma = calculate_wma(raw_hma, sqrt_window)
        
        # Pad to original length
        hma_padded = np.full(len(values), np.nan)
        start_idx = window - half_window - sqrt_window + 1
        end_idx = start_idx + len(hma)
        hma_padded[start_idx:end_idx] = hma
        
        return hma_padded
    
    hma_21_1d = calculate_hma(close_1d, 21)
    
    # Align 1d HMA to 4h timeframe
    hma_21_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_21_1d)
    
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
        # Volume confirmation: volume > 2.0x 50-period average (strict to reduce trades)
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
        curr_atr = atr[i]
        curr_upper = donchian_upper_aligned[i]
        curr_lower = donchian_lower_aligned[i]
        curr_hma = hma_21_1d_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike and trend alignment
            if volume_spike and not np.isnan(curr_hma):
                # Bullish entry: price breaks above 1d Donchian upper with 1d uptrend (close > HMA21)
                if curr_close > curr_upper and curr_close > curr_hma:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish entry: price breaks below 1d Donchian lower with 1d downtrend (close < HMA21)
                elif curr_close < curr_lower and curr_close < curr_hma:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Stoploss: 2.0 * ATR below entry price OR price breaks 1d Donchian lower (reversal signal)
            if curr_close < entry_price - 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            elif curr_close < curr_lower:
                signals[i] = 0.0
                position = 0
            # Take profit: price reaches 1.5x ATR above entry OR touches 1d Donchian upper (mean reversion)
            elif curr_close > entry_price + 1.5 * curr_atr:
                signals[i] = 0.10  # reduce position
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: 2.0 * ATR above entry price OR price breaks 1d Donchian upper (reversal signal)
            if curr_close > entry_price + 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            elif curr_close > curr_upper:
                signals[i] = 0.0
                position = 0
            # Take profit: price reaches 1.5x ATR below entry OR touches 1d Donchian lower (mean reversion)
            elif curr_close < entry_price - 1.5 * curr_atr:
                signals[i] = -0.10  # reduce position
            else:
                signals[i] = -0.25
    
    return signals