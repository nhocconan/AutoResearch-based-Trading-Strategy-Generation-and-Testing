#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d HMA21 trend filter and volume confirmation.
# Long when price breaks above Donchian upper band AND close > 1d HMA21 AND volume > 1.5x 20-period volume median.
# Short when price breaks below Donchian lower band AND close < 1d HMA21 AND volume > 1.5x 20-period volume median.
# Uses discrete sizing 0.25. ATR(14) stoploss: signal→0 when price moves against position by 2.0*ATR.
# Donchian provides structure, HMA21 filters trend, volume confirms breakout strength.
# Target: 19-50 trades/year on 4h timeframe (75-200 total over 4 years).

name = "4h_Donchian20_1dHMA21_Volume_Breakout_v1"
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
    
    # Calculate ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr_first = np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])
    tr = np.concatenate([[tr_first], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 20-period volume median for volume confirmation
    vol_median_20 = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    
    # Calculate Donchian(20) channels (using prior bar's data to avoid look-ahead)
    prev_high = np.concatenate([[high[0]], high[:-1]])
    prev_low = np.concatenate([[low[0]], low[:-1]])
    donchian_upper = pd.Series(prev_high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(prev_low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d HMA21 trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 21:
        return np.zeros(n)
    
    # Hull Moving Average calculation
    half_length = 21 // 2
    sqrt_length = int(np.sqrt(21))
    
    wma_half = pd.Series(df_1d['close'].values).rolling(window=half_length, min_periods=half_length).apply(
        lambda x: np.average(x, weights=np.arange(1, half_length + 1)), raw=True
    ).values
    wma_full = pd.Series(df_1d['close'].values).rolling(window=21, min_periods=21).apply(
        lambda x: np.average(x, weights=np.arange(1, 22)), raw=True
    ).values
    wma_sqrt = pd.Series((2 * wma_half - wma_full)).rolling(window=sqrt_length, min_periods=sqrt_length).apply(
        lambda x: np.average(x, weights=np.arange(1, sqrt_length + 1)), raw=True
    ).values
    hma_21_1d = wma_sqrt
    
    hma_21_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_21_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # track entry price for stoploss
    
    # Start after warmup for Donchian, HMA, ATR, volume
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(atr[i]) or 
            np.isnan(hma_21_1d_aligned[i]) or 
            np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or 
            np.isnan(vol_median_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Trend filter: price vs 1d HMA21
        uptrend = curr_close > hma_21_1d_aligned[i]
        downtrend = curr_close < hma_21_1d_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 20-period volume median
        if vol_median_20[i] <= 0 or np.isnan(vol_median_20[i]):
            volume_confirm = False
        else:
            volume_confirm = curr_volume > (vol_median_20[i] * 1.5)
        
        if position == 0:  # Flat - look for new entries
            # Long: price > Donchian upper AND uptrend AND volume spike
            if curr_close > donchian_upper[i] and uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: price < Donchian lower AND downtrend AND volume spike
            elif curr_close < donchian_lower[i] and downtrend and volume_confirm:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Stoploss: price moves against position by 2.0*ATR
            if curr_close < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price breaks below Donchian lower OR trend turns down
            elif curr_close < donchian_lower[i] or not uptrend:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: price moves against position by 2.0*ATR
            if curr_close > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price breaks above Donchian upper OR trend turns up
            elif curr_close > donchian_upper[i] or not downtrend:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
    
    return signals