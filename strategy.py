#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 1d for EMA50 trend filter (avoids counter-trend trades in bear markets).
- Entry: Long when price breaks above Donchian upper(20) AND price > 1d EMA50 AND volume > 2.0 * 4h volume MA(20);
         Short when price breaks below Donchian lower(20) AND price < 1d EMA50 AND volume > 2.0 * 4h volume MA(20).
- Exit: ATR-based trailing stop (signal=0 when long and price < highest_high - 2.5*ATR, or short and price > lowest_low + 2.5*ATR).
- Signal size: 0.30 discrete to balance profit potential and fee drag.
- Donchian channels provide clear structure; 1d EMA50 filters weak breakouts; volume confirms conviction.
- Designed to work in both bull (trend continuation) and bear (mean reversion via shorts) markets.
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
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate EMA(50) on 1d
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Get 4h data for Donchian channels and volume MA
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    volume_4h = df_4h['volume'].values
    
    # Calculate Donchian channels (20-period) on 4h
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Calculate ATR(14) on 4h for trailing stop
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate volume MA(20) on 4h
    vol_ma_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to primary 4h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    atr_14_aligned = align_htf_to_ltf(prices, df_4h, atr_14)
    vol_ma_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_high = 0.0
    lowest_low = 0.0
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(atr_14_aligned[i]) or 
            np.isnan(vol_ma_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Update highest/lowest for trailing stop
        if position == 1:
            highest_high = max(highest_high, curr_high)
        elif position == -1:
            lowest_low = min(lowest_low, curr_low)
        
        # ATR-based trailing stop
        if position == 1:
            if curr_close < highest_high - 2.5 * atr_14_aligned[i]:
                signals[i] = 0.0
                position = 0
                highest_high = 0.0
                continue
        elif position == -1:
            if curr_close > lowest_low + 2.5 * atr_14_aligned[i]:
                signals[i] = 0.0
                position = 0
                lowest_low = 0.0
                continue
        
        # Breakout conditions with volume confirmation and trend filter
        bullish_breakout = curr_close > donchian_high_aligned[i]
        bearish_breakout = curr_close < donchian_low_aligned[i]
        
        # Trend filter from 1d EMA50
        price_above_ema = curr_close > ema_50_aligned[i]
        price_below_ema = curr_close < ema_50_aligned[i]
        
        # Volume confirmation (strong breakout requires high volume)
        vol_confirm = curr_volume > 2.0 * vol_ma_aligned[i]
        
        if position == 0:
            # Check for entry signals
            if vol_confirm:
                # Long: bullish breakout AND price above 1d EMA50
                if bullish_breakout and price_above_ema:
                    signals[i] = 0.30
                    position = 1
                    entry_price = curr_close
                    highest_high = curr_high
                # Short: bearish breakout AND price below 1d EMA50
                elif bearish_breakout and price_below_ema:
                    signals[i] = -0.30
                    position = -1
                    entry_price = curr_close
                    lowest_low = curr_low
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.30
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.30
    
    return signals

name = "4h_Donchian20_1dEMA50_VolumeConfirm_ATRStop_v1"
timeframe = "4h"
leverage = 1.0