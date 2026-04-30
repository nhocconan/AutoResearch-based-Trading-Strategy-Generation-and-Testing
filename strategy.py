#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1w Donchian(20) breakout with 1d volume confirmation and ATR stoploss
# Uses weekly structure for major trend, 1d volume for institutional participation confirmation
# Designed for very low trade frequency (<25/year) to minimize fee drag in both bull and bear markets
# Weekly Donchian breakouts capture major trend changes with high follow-through
# Volume confirmation filters false breakouts
# ATR-based stoploss manages risk without look-ahead bias

name = "12h_Donchian20_1wTrend_1dVolumeSpike_ATRStop_v1"
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
    
    # Load weekly data ONCE before loop for Donchian calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Load daily data ONCE before loop for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate weekly Donchian channels (20-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate rolling max/min for Donchian upper/lower bands
    high_series = pd.Series(high_1w)
    low_series = pd.Series(low_1w)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Align weekly Donchian levels to 12h timeframe (wait for completed weekly bar)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1w, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1w, donchian_lower)
    
    # Calculate daily volume average for confirmation (20-period)
    volume_1d = df_1d['volume'].values
    volume_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ma_20_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20)
    
    # Calculate ATR(14) for dynamic stoploss on 12h data
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 50  # warmup for ATR and indicators
    
    for i in range(start_idx, n):
        # Volume confirmation: current 12h volume > 1.5x daily average volume
        vol_ma_20_curr = volume_ma_20_aligned[i]
        volume_confirmed = volume[i] > (1.5 * vol_ma_20_curr)
        
        curr_close = close[i]
        curr_atr = atr[i]
        curr_dc_upper = donchian_upper_aligned[i]
        curr_dc_lower = donchian_lower_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume confirmation for breakout validity
            if volume_confirmed:
                # Bullish entry: price breaks above weekly Donchian upper band
                if curr_close > curr_dc_upper:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish entry: price breaks below weekly Donchian lower band
                elif curr_close < curr_dc_lower:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Stoploss: 2.0 * ATR below entry price
            if curr_close < entry_price - 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            # Exit: price re-enters weekly Donchian channel (mean reversion signal)
            elif curr_close < curr_dc_upper and curr_close > curr_dc_lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: 2.0 * ATR above entry price
            if curr_close > entry_price + 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            # Exit: price re-enters weekly Donchian channel (mean reversion signal)
            elif curr_close < curr_dc_upper and curr_close > curr_dc_lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals