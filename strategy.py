#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with 1w EMA34 trend filter and volume spike confirmation.
- Primary timeframe: 1d targeting 30-100 total trades over 4 years (7-25/year).
- HTF: 1w EMA34 for trend direction (bullish when close > EMA34, bearish when close < EMA34).
- Entry: Long when price breaks above Donchian(20) upper band in 1w bull trend with volume > 2.0 * 1d volume MA(20).
         Short when price breaks below Donchian(20) lower band in 1w bear trend with volume > 2.0 * 1d volume MA(20).
- Exit: Opposite Donchian(10) breakout or ATR-based stoploss (2.5 * ATR(20)).
- Signal size: 0.25 discrete to balance capture and fee control.
- Designed for BTC/ETH: Donchian breakouts capture trending moves, 1w EMA34 filter avoids counter-trend trades,
  volume confirmation ensures conviction, works in both bull and bear markets via directional filtering.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian channels and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Get 1w data for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 1d Donchian(20) upper and lower bands
    highest_high_20 = pd.Series(df_1d['high']).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(df_1d['low']).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d Donchian(10) for exit (opposite band)
    highest_high_10 = pd.Series(df_1d['high']).rolling(window=10, min_periods=10).max().values
    lowest_low_10 = pd.Series(df_1d['low']).rolling(window=10, min_periods=10).min().values
    
    # Align Donchian bands from 1d to 1d timeframe (direct use with alignment for safety)
    upper_20_aligned = align_htf_to_ltf(prices, df_1d, highest_high_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_1d, lowest_low_20)
    upper_10_aligned = align_htf_to_ltf(prices, df_1d, highest_high_10)
    lower_10_aligned = align_htf_to_ltf(prices, df_1d, lowest_low_10)
    
    # Calculate 1d ATR(20) for stoploss
    tr1 = pd.Series(df_1d['high'] - df_1d['low']).values
    tr2 = pd.Series(np.abs(df_1d['high'] - df_1d['close'].shift(1))).fillna(0).values
    tr3 = pd.Series(np.abs(df_1d['low'] - df_1d['close'].shift(1))).fillna(0).values
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_20 = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    atr_20_aligned = align_htf_to_ltf(prices, df_1d, atr_20)
    
    # Calculate 1d volume MA(20) for confirmation
    volume_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Calculate 1w EMA34 for trend
    ema_34 = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1w, ema_34)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    atr_stop = 0.0
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(upper_20_aligned[i]) or np.isnan(lower_20_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]) or
            np.isnan(atr_20_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Volume confirmation: 2.0x threshold
        vol_confirmed = curr_volume > 2.0 * vol_ma_1d_aligned[i]
        
        # Determine 1w EMA34 trend: bullish if close > EMA34, bearish if close < EMA34
        trend_bullish = close[i] > ema_34_aligned[i]
        trend_bearish = close[i] < ema_34_aligned[i]
        
        if position == 0:
            # Check for entry signals
            # Long: Price breaks above Donchian(20) upper band in 1w bull trend with volume confirmation
            if curr_high > upper_20_aligned[i] and trend_bullish and vol_confirmed:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
                atr_stop = entry_price - 2.5 * atr_20_aligned[i]
            # Short: Price breaks below Donchian(20) lower band in 1w bear trend with volume confirmation
            elif curr_low < lower_20_aligned[i] and trend_bearish and vol_confirmed:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
                atr_stop = entry_price + 2.5 * atr_20_aligned[i]
        elif position == 1:
            # Long position: exit on Donchian(10) lower band break or ATR stoploss
            if curr_low < lower_10_aligned[i] or curr_low <= atr_stop:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                # Update trailing stop (optional - keep initial stop for simplicity)
        elif position == -1:
            # Short position: exit on Donchian(10) upper band break or ATR stoploss
            if curr_high > upper_10_aligned[i] or curr_high >= atr_stop:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_1wEMA34_Trend_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0