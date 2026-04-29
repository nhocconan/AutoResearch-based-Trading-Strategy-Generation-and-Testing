#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d trend filter and volume confirmation
# Donchian breakouts capture strong momentum moves in both bull and bear markets
# 1d EMA50 trend filter ensures we trade in direction of higher timeframe trend
# Volume confirmation validates breakout strength and reduces false signals
# ATR-based stoploss manages risk during adverse moves
# Target: 12-37 trades/year (50-150 total over 4 years) for 12h timeframe

name = "12h_Donchian20_1dTrend_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop for daily calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Calculate daily EMA50 for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema50_1d = close_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate Donchian Channel (20-period) on 12h data
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate ATR (14) for stoploss
    tr1 = pd.Series(high - low).values
    tr2 = pd.Series(np.abs(high - np.roll(close, 1))).values
    tr3 = pd.Series(np.abs(low - np.roll(close, 1))).values
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(60, 50, 20, 14)  # warmup for all indicators
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if np.isnan(ema50_1d_aligned[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(atr[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_ema50_1d = ema50_1d_aligned[i]
        curr_atr = atr[i]
        
        # Volume confirmation: volume > 1.5x 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-20:i])
            volume_confirm = curr_volume > (1.5 * vol_ma_20)
        else:
            volume_confirm = False
        
        if position == 0:  # Flat - look for new entries
            # Bullish breakout: price breaks above Donchian upper band in bullish regime
            if curr_close > highest_high[i] and curr_close > curr_ema50_1d and volume_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Bearish breakout: price breaks below Donchian lower band in bearish regime
            elif curr_close < lowest_low[i] and curr_close < curr_ema50_1d and volume_confirm:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        
        elif position == 1:  # Long position - exit conditions
            # Stoploss: 2.0 * ATR below entry price
            if curr_close < entry_price - 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            # Exit when price crosses below Donchian middle (mean of bands)
            elif curr_close < (highest_high[i] + lowest_low[i]) / 2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position - exit conditions
            # Stoploss: 2.0 * ATR above entry price
            if curr_close > entry_price + 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            # Exit when price crosses above Donchian middle
            elif curr_close > (highest_high[i] + lowest_low[i]) / 2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals