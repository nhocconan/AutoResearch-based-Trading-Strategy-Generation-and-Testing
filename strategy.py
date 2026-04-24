#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter, volume spike confirmation, and ATR-based trailing stop.
- Primary timeframe: 1d targeting 30-100 total trades over 4 years (7-25/year).
- HTF: 1w for EMA50 trend filter (price above/below weekly EMA).
- Entry: Long when price breaks above Donchian(20) high AND price > 1w EMA50 AND volume > 2.0 * 1d volume MA(20);
         Short when price breaks below Donchian(20) low AND price < 1w EMA50 AND volume > 2.0 * 1d volume MA(20).
- Exit: ATR(14) trailing stop (long: highest_high - 2.5*ATR; short: lowest_low + 2.5*ATR) or opposite Donchian breakout.
- Signal size: 0.25 discrete to balance profit potential and fee drag.
- Designed to capture strong trends with volatility-adjusted exits and volume confirmation to avoid false breakouts.
- Donchian channels provide structured support/resistance that works in both ranging and trending markets.
- Weekly EMA50 ensures we only trade with the dominant long-term trend, reducing whipsaw in bear markets.
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
    
    # Get 1d data for Donchian and volume MA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Donchian(20) on 1d using previous 20 periods
    # Donchian high = max(high over last 20 periods)
    # Donchian low = min(low over last 20 periods)
    high_series = pd.Series(high_1d)
    low_series = pd.Series(low_1d)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().shift(1).values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().shift(1).values
    
    # Calculate ATR(14) on 1d for stoploss
    tr1 = pd.Series(high_1d - low_1d)
    tr2 = pd.Series(np.abs(high_1d - np.roll(close_1d, 1)))
    tr3 = pd.Series(np.abs(low_1d - np.roll(close_1d, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    tr.iloc[0] = high_1d[0] - low_1d[0]  # First value
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate volume MA(20) on 1d
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 55:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Calculate EMA(50) on 1w
    ema_50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all indicators to primary 1d timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr)
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    # Track extreme prices for trailing stop
    highest_high = 0.0
    lowest_low = float('inf')
    
    # Start from index where all indicators are ready (max of 20 for Donchian, 20 for volume, 14 for ATR, 50 for EMA)
    start_idx = max(20, 20, 14, 50)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(vol_ma_aligned[i]) or np.isnan(atr_aligned[i]) or np.isnan(ema_50_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                highest_high = 0.0
                lowest_low = float('inf')
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Update trailing stop extremes
        if position == 1:  # long
            if curr_high > highest_high:
                highest_high = curr_high
        elif position == -1:  # short
            if curr_low < lowest_low:
                lowest_low = curr_low
        
        # Calculate stop levels
        long_stop = highest_high - 2.5 * atr_aligned[i] if highest_high > 0 else 0.0
        short_stop = lowest_low + 2.5 * atr_aligned[i] if lowest_low != float('inf') else float('inf')
        
        # Check for stoploss
        if position == 1 and curr_close < long_stop:
            signals[i] = 0.0
            position = 0
            highest_high = 0.0
            lowest_low = float('inf')
            continue
        elif position == -1 and curr_close > short_stop:
            signals[i] = 0.0
            position = 0
            highest_high = 0.0
            lowest_low = float('inf')
            continue
        
        # Breakout conditions
        bullish_breakout = curr_close > donchian_high_aligned[i]
        bearish_breakout = curr_close < donchian_low_aligned[i]
        
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
                    highest_high = curr_high
                    lowest_low = float('inf')
                # Short: bearish breakout AND price below 1w EMA50
                elif bearish_breakout and price_below_ema:
                    signals[i] = -0.25
                    position = -1
                    highest_high = 0.0
                    lowest_low = curr_low
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_Breakout_1wEMA50_Trend_VolumeSpike_ATRStop_v1"
timeframe = "1d"
leverage = 1.0