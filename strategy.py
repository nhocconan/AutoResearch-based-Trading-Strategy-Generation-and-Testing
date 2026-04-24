#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d ATR filter and volume confirmation.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 1d ATR(14) for volatility filter - only trade when ATR > 1.5 * 20-period MA of ATR (high volatility regime).
- Entry: Price breaks above/below 4h Donchian(20) channel with volume > 1.5 * 4h volume MA(20) and ATR filter active.
- Exit: ATR-based trailing stop (3 * ATR from extreme) or opposite Donchian break.
- Signal size: 0.25 discrete for fee control.
- Designed for BTC/ETH: Donchian captures breakouts, ATR filter avoids low-volatility chop, volume confirms strength.
- Works in bull markets by catching trends, works in bear markets by fading failed breakouts during high volatility.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Donchian levels and volume
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Get 1d data for ATR filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # need enough for ATR calculation
        return np.zeros(n)
    
    # Calculate 4h Donchian(20) channels
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Donchian upper = max(high, lookback=20), lower = min(low, lookback=20)
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels from 4h to lower timeframe (prices timeframe)
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    # Calculate 4h volume MA(20) for confirmation
    volume_4h = df_4h['volume'].values
    vol_ma_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    vol_ma_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_4h)
    
    # Calculate 1d ATR(14) for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    # Set first TR to high-low (no previous close)
    tr[0] = tr1[0]
    
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma_20 = pd.Series(atr_14).rolling(window=20, min_periods=20).mean().values
    atr_filter = atr_14 > (1.5 * atr_ma_20)  # High volatility regime
    
    # Align ATR filter from 1d to lower timeframe
    atr_filter_aligned = align_htf_to_ltf(prices, df_1d, atr_filter)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    long_stop = 0.0
    short_stop = 0.0
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(vol_ma_4h_aligned[i]) or np.isnan(atr_filter_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        if position == 0:
            # Check for entry signals with volume confirmation and ATR filter
            vol_confirmed = curr_volume > 1.5 * vol_ma_4h_aligned[i]
            atr_active = bool(atr_filter_aligned[i])
            
            # Long: price breaks above Donchian upper AND volume confirmed AND ATR filter active
            if curr_high > donchian_high_aligned[i] and vol_confirmed and atr_active:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
                long_stop = curr_close - 3.0 * atr_14[-1] if len(atr_14) > 0 else curr_close * 0.97  # approximate
            # Short: price breaks below Donchian lower AND volume confirmed AND ATR filter active
            elif curr_low < donchian_low_aligned[i] and vol_confirmed and atr_active:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
                short_stop = curr_close + 3.0 * atr_14[-1] if len(atr_14) > 0 else curr_close * 1.03  # approximate
        elif position == 1:
            # Long position: update ATR-based trailing stop and check exit
            # Update stop to trail 3*ATR below highest high since entry
            # Since we can't track intrabar, use close-based approximation
            long_stop = max(long_stop, curr_close - 3.0 * atr_14[-1] if len(atr_14) > 0 else long_stop)
            if curr_low < long_stop or curr_low < donchian_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: update ATR-based trailing stop and check exit
            # Update stop to trail 3*ATR above lowest low since entry
            short_stop = min(short_stop, curr_close + 3.0 * atr_14[-1] if len(atr_14) > 0 else short_stop)
            if curr_high > short_stop or curr_high > donchian_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_1dATR_VolumeFilter_v1"
timeframe = "4h"
leverage = 1.0