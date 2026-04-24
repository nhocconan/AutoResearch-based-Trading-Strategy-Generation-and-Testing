#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout + 1d ATR regime filter + volume spike confirmation.
- Primary timeframe: 12h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d ATR(14) for volatility regime (high volatility = ATR > 1.5 * 50-period ATR MA).
- Donchian channels: calculated from prior 12h OHLC; long on break above upper band, short on breakdown below lower band.
- Volume confirmation: current volume > 2.0 * 20-period volume MA to filter weak signals.
- ATR-based stoploss: exit when price moves against position by 2.5 * ATR(14) (using 12h ATR).
- Signal size: 0.25 discrete to minimize fee churn while maintaining profitability.
Designed to capture volatility expansion breakouts with proper filtering to avoid overtrading and fee drag.
Works in both bull and bear markets by using volatility regime filter and volatility-based stops.
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
    
    # Get 1d data for ATR regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need for ATR MA
        return np.zeros(n)
    
    # Calculate 1d ATR(14) and its 50-period MA for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma_50 = pd.Series(atr_1d).rolling(window=50, min_periods=50).mean().values
    atr_regime = align_htf_to_ltf(prices, df_1d, atr_ma_50)
    atr_current = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate 12h ATR(14) for stoploss
    tr1_12h = high - low
    tr2_12h = np.abs(high - np.roll(close, 1))
    tr3_12h = np.abs(low - np.roll(close, 1))
    tr2_12h[0] = 0
    tr3_12h[0] = 0
    tr_12h = np.maximum(tr1_12h, np.maximum(tr2_12h, tr3_12h))
    atr_12h = pd.Series(tr_12h).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 12h volume MA(20) for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Donchian channels from prior 12h OHLC (20-period)
    # Need to calculate from price series directly since we don't have HTF 12h data
    # We'll calculate rolling max/min on the 12h timeframe data
    # Since we're already on 12h timeframe, we can use the current prices directly
    donchian_upper = pd.Series(close).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(close).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to prevent look-ahead (use previous bar's values)
    donchian_upper_aligned = np.roll(donchian_upper, 1)
    donchian_lower_aligned = np.roll(donchian_lower, 1)
    donchian_upper_aligned[0] = np.nan
    donchian_lower_aligned[0] = np.nan
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from index where all indicators are ready
    start_idx = max(100, 50, 20, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(atr_regime[i]) or np.isnan(atr_current[i]) or 
            np.isnan(atr_12h[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Volatility regime filter: only trade when current ATR > 1.5 * ATR MA (high volatility)
        vol_regime = atr_current[i] > 1.5 * atr_regime[i]
        
        if position == 0:
            # Check for entry signals with volume confirmation (2.0x threshold) and volatility regime
            vol_confirmed = curr_volume > 2.0 * vol_ma[i]
            
            # Long: price breaks above Donchian upper band AND volatility regime AND volume confirmed
            if curr_high > donchian_upper_aligned[i] and vol_regime and vol_confirmed:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: price breaks below Donchian lower band AND volatility regime AND volume confirmed
            elif curr_low < donchian_lower_aligned[i] and vol_regime and vol_confirmed:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        elif position == 1:
            # Long position: exit on stoploss (2.5 * ATR) or price breaks below Donchian lower band
            stop_loss = entry_price - 2.5 * atr_12h[i]
            if curr_low < stop_loss or curr_low < donchian_lower_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit on stoploss (2.5 * ATR) or price breaks above Donchian upper band
            stop_loss = entry_price + 2.5 * atr_12h[i]
            if curr_high > stop_loss or curr_high > donchian_upper_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_1dATR_Regime_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0