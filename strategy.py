#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian channel breakout with 1d ATR filter and volume confirmation.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 1d for ATR-based volatility filter.
- Entry: Long when price breaks above Donchian upper(20) AND 1d ATR(14) > 1d ATR MA(50) AND volume > 1.3 * 4h volume MA(20);
         Short when price breaks below Donchian lower(20) AND 1d ATR(14) > 1d ATR MA(50) AND volume > 1.3 * 4h volume MA(20).
- Exit: Close-based reversal (opposite signal) or volatility contraction (signal=0 when 1d ATR(14) < 1d ATR MA(50)).
- Signal size: 0.25 discrete to minimize fee drag while maintaining profit potential.
- Donchian channels provide objective breakout levels; ATR filter ensures we only trade during sufficient volatility (avoids choppy/low-vol environments);
  volume confirmation avoids false breakouts. Works in bull markets (buy breakouts) and bear markets (sell breakdowns) with volatility filter.
- Estimated trades: ~100 total over 4 years (~25/year) based on Donchian breakout frequency with filters.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 4h Donchian channels (20-period)
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 4h volume MA(20) for confirmation
    vol_ma_4h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d ATR(14) for volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # First bar: no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 1d ATR MA(50) for volatility regime filter
    atr_ma_50 = pd.Series(atr_14).rolling(window=50, min_periods=50).mean().values
    
    # Align all indicators to primary 4h timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, prices, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, prices, donchian_lower)
    vol_ma_4h_aligned = align_htf_to_ltf(prices, prices, vol_ma_4h)
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    atr_ma_50_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 60  # Need sufficient data for Donchian(20), volume MA(20), ATR(14), ATR MA(50)
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(vol_ma_4h_aligned[i]) or np.isnan(atr_14_aligned[i]) or np.isnan(atr_ma_50_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Exit: volatility contraction (ATR < ATR MA) or opposite breakout
        if position != 0:
            vol_contract = atr_14_aligned[i] < atr_ma_50_aligned[i]
            if vol_contract:
                signals[i] = 0.0
                position = 0
                continue
            
            # Opposite breakout exit
            if position == 1 and curr_low < donchian_lower_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            elif position == -1 and curr_high > donchian_upper_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
        
        # Entry conditions: Donchian breakout + volatility expansion + volume confirmation
        bullish_breakout = curr_high > donchian_upper_aligned[i]  # Break above upper band
        bearish_breakout = curr_low < donchian_lower_aligned[i]   # Break below lower band
        
        # Volatility filter: only trade when ATR > ATR MA (expanding volatility)
        vol_expanding = atr_14_aligned[i] > atr_ma_50_aligned[i]
        
        # Volume confirmation (1.3x average volume)
        vol_confirm = curr_volume > 1.3 * vol_ma_4h_aligned[i]
        
        if position == 0:
            # Check for entry signals
            if vol_expanding and vol_confirm:
                # Long: Price breaks above Donchian upper AND volatility expanding
                if bullish_breakout:
                    signals[i] = 0.25
                    position = 1
                # Short: Price breaks below Donchian lower AND volatility expanding
                elif bearish_breakout:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_ATR14_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0