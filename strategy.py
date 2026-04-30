#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d HMA(21) trend filter and volume confirmation
# Donchian channels provide robust volatility-based support/resistance that works in both bull and bear markets.
# Breakouts above/below 20-period Donchian with volume spike indicate genuine momentum.
# 1d HMA(21) filters trades to align with higher-timeframe trend, reducing false breakouts.
# Designed for moderate trade frequency (~30-60/year on 4h) to balance opportunity and fee drag.

name = "4h_Donchian20_Breakout_1dHMA21_VolumeSpike_v1"
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
    
    # Load 1d data ONCE before loop for HMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 21:
        return np.zeros(n)
    
    # Calculate 1d HMA(21) for trend filter
    # HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    half_len = 21 // 2
    sqrt_len = int(np.sqrt(21))
    
    def wma(values, window):
        weights = np.arange(1, window + 1)
        return np.convolve(values, weights, 'valid') / weights.sum()
    
    close_1d = df_1d['close'].values
    wma_half = wma(close_1d, half_len)
    wma_full = wma(close_1d, 21)
    hma_raw = 2 * wma_half - wma_full
    hma_1d = wma(hma_raw, sqrt_len)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate ATR(14) for dynamic stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 20  # warmup for Donchian
    
    for i in range(start_idx, n):
        # Volume confirmation: volume > 1.5x 20-period average
        vol_ma_20 = np.mean(volume[max(0, i-20):i]) if i >= 20 else np.mean(volume[:i]) if i > 0 else 0
        volume_spike = volume[i] > (1.5 * vol_ma_20) if i > 0 else False
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema = hma_1d_aligned[i]
        curr_atr = atr[i]
        
        # Calculate Donchian channels (20-period)
        donchian_high = np.max(high[max(0, i-19):i+1])
        donchian_low = np.min(low[max(0, i-19):i+1])
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike and trend alignment
            if volume_spike:
                # Bullish entry: price breaks above Donchian high with 1d uptrend
                if curr_close > donchian_high and curr_close > curr_ema:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish entry: price breaks below Donchian low with 1d downtrend
                elif curr_close < donchian_low and curr_close < curr_ema:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Stoploss: 2.0 * ATR below entry price OR price breaks Donchian low
            if curr_close < entry_price - 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            elif curr_close < donchian_low:
                signals[i] = 0.0
                position = 0
            # Take profit: price reaches Donchian high (mean reversion tendency)
            elif curr_close >= donchian_high:
                signals[i] = 0.10  # reduce position
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: 2.0 * ATR above entry price OR price breaks Donchian high
            if curr_close > entry_price + 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            elif curr_close > donchian_high:
                signals[i] = 0.0
                position = 0
            # Take profit: price reaches Donchian low (mean reversion tendency)
            elif curr_close <= donchian_low:
                signals[i] = -0.10  # reduce position
            else:
                signals[i] = -0.25
    
    return signals