#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation.
- Primary timeframe: 1d targeting 30-100 total trades over 4 years (7-25/year).
- HTF: 1w for EMA50 trend filter (calculated from prior 1w bar).
- Entry: Long when price breaks above Donchian(20) high AND price > 1w EMA50 AND volume > 1.5 * 1d volume MA(20);
         Short when price breaks below Donchian(20) low AND price < 1w EMA50 AND volume > 1.5 * 1d volume MA(20).
- Exit: Close-based reversal (opposite signal) or ATR-based stoploss (signal=0 when price closes below/above ATR-based level).
- Signal size: 0.25 discrete to minimize fee drag while maintaining profit potential.
- Donchian channels provide structural breakouts; 1w EMA50 filters counter-trend moves in bear markets.
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
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate EMA(50) on 1w
    ema_50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Donchian(20) on 1d (using prior 20 bars to avoid look-ahead)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().shift(1).values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().shift(1).values
    
    # Calculate ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar has no previous close
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Get 1d data for volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    
    # Align 1w EMA50 to 1d timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready (max of 50 for EMA, 20 for Donchian/volume, 14 for ATR)
    start_idx = max(50, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # ATR-based stoploss: exit if price closes 2*ATR against position
        if position == 1:
            if curr_close < (donchian_high[i] - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:
            if curr_close > (donchian_low[i] + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                continue
        
        # Breakout conditions with volume confirmation and trend filter
        bullish_breakout = curr_close > donchian_high[i]
        bearish_breakout = curr_close < donchian_low[i]
        
        # Trend filter from 1w EMA50
        price_above_ema = curr_close > ema_50_aligned[i]
        price_below_ema = curr_close < ema_50_aligned[i]
        
        # Volume confirmation
        vol_confirm = curr_volume > 1.5 * volume_ma[i]
        
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

name = "1d_Donchian20_Breakout_1wEMA50_Trend_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0