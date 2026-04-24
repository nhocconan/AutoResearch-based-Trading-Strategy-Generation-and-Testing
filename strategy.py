#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 1d for EMA34 trend filter.
- Entry: Long when price breaks above Donchian(20) high AND price > 1d EMA34 AND volume > 1.5 * 4h volume MA(20);
         Short when price breaks below Donchian(20) low AND price < 1d EMA34 AND volume > 1.5 * 4h volume MA(20).
- Exit: Close-based reversal (opposite signal) or stoploss via trend filter (signal=0 when price closes below/above 1d EMA34).
- Signal size: 0.30 discrete to balance profit potential and fee drag.
- Donchian channels provide structural breakout levels; 1d EMA34 filters counter-trend breakouts.
- Works in bull markets via breakout continuation and bear markets via trend-filtered mean reversion at channel levels.
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
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate EMA(34) on 1d
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Get 4h data for volume MA
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    volume_4h = df_4h['volume'].values
    
    # Calculate volume MA(20) on 4h
    vol_ma_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Donchian(20) channels on primary 4h data
    # Donchian high: highest high over last 20 periods
    # Donchian low: lowest low over last 20 periods
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Align all indicators to primary 4h timeframe
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    vol_ma_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 34  # EMA34 needs 34 periods
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma_aligned[i]) or 
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Stoploss: exit if price closes below/above 1d EMA34 (trend filter)
        if position == 1:
            if curr_close < ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:
            if curr_close > ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
        
        # Breakout conditions with volume confirmation and trend filter
        bullish_breakout = curr_high > donchian_high[i]
        bearish_breakout = curr_low < donchian_low[i]
        
        # Trend filter from 1d EMA34
        price_above_ema = curr_close > ema_34_aligned[i]
        price_below_ema = curr_close < ema_34_aligned[i]
        
        # Volume confirmation
        vol_confirm = curr_volume > 1.5 * vol_ma_aligned[i]
        
        if position == 0:
            # Check for entry signals
            if vol_confirm:
                # Long: bullish breakout above Donchian high AND price above 1d EMA34
                if bullish_breakout and price_above_ema:
                    signals[i] = 0.30
                    position = 1
                # Short: bearish breakout below Donchian low AND price below 1d EMA34
                elif bearish_breakout and price_below_ema:
                    signals[i] = -0.30
                    position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.30
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.30
    
    return signals

name = "4h_Donchian20_1dEMA34_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0