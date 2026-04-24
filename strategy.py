#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation.
- Primary timeframe: 1d targeting 30-100 total trades over 4 years (7-25/year).
- HTF: 1w for EMA50 trend filter (more stable for BTC/ETH trend identification).
- Entry: Long when price breaks above Donchian(20) high AND price > 1w EMA50 AND volume > 2.0 * 1d volume MA(20);
         Short when price breaks below Donchian(20) low AND price < 1w EMA50 AND volume > 2.0 * 1d volume MA(20).
- Exit: Close-based reversal (opposite signal) or stoploss via Donchian(10) opposite channel.
- Signal size: 0.25 discrete to minimize fee drag while maintaining profit potential.
- Donchian channels provide clear structure; 1w EMA50 filters counter-trend signals; volume confirmation avoids false breakouts.
- Works in bull markets (buy strength) and bear markets (sell weakness) with volume confirmation.
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
    
    # Calculate Donchian(20) channels (primary timeframe)
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    donch_high_20 = high_s.rolling(window=20, min_periods=20).max().values
    donch_low_20 = low_s.rolling(window=20, min_periods=20).min().values
    
    # Calculate Donchian(10) for stoploss (opposite channel)
    donch_high_10 = high_s.rolling(window=10, min_periods=10).max().values
    donch_low_10 = low_s.rolling(window=10, min_periods=10).min().values
    
    # Get 1w data for EMA50
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate EMA(50) on 1w
    ema_50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Get 1d data for volume MA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    
    # Calculate volume MA(20) on 1d
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to primary 1d timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50, 20)  # Donchian20, EMA50, volume MA20
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(donch_high_20[i]) or np.isnan(donch_low_20[i]) or 
            np.isnan(donch_high_10[i]) or np.isnan(donch_low_10[i]) or
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Stoploss: exit if price closes below/above Donchian(10) opposite channel
        if position == 1:
            if curr_close < donch_low_10[i]:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:
            if curr_close > donch_high_10[i]:
                signals[i] = 0.0
                position = 0
                continue
        
        # Entry conditions with volume confirmation and trend filter
        bullish_breakout = curr_high > donch_high_20[i]
        bearish_breakout = curr_low < donch_low_20[i]
        
        # Trend filter from 1w EMA50
        price_above_ema = curr_close > ema_50_aligned[i]
        price_below_ema = curr_close < ema_50_aligned[i]
        
        # Volume confirmation
        vol_confirm = curr_volume > 2.0 * vol_ma_aligned[i]
        
        if position == 0:
            # Check for entry signals
            if vol_confirm:
                # Long: bullish breakout AND price above 1w EMA50
                if bullish_breakout and price_above_ema:
                    signals[i] = 0.25
                    position = 1
                # Short: bearish breakout AND price below 1w EMA50
                elif bearish_breakout and price_below_ema:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_1wEMA50_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0