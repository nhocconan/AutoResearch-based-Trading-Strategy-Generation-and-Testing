#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout + 1w EMA50 trend filter + volume confirmation + ATR stoploss
- Donchian breakout on daily captures strong multi-day moves
- 1w EMA50 filters for weekly trend alignment to avoid counter-trend trades
- Volume spike (>1.5x 20-day average) confirms breakout strength
- ATR-based stoploss (2.0 * ATR) manages risk
- Target: 15-25 trades/year per symbol (~60-100 total over 4 years)
- Position sizing: 0.25 (discrete levels to minimize fee churn)
- Works in both bull and bear markets: breakouts capture strong moves, trend filter avoids whipsaws in ranging/bear conditions
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for primary calculations (Donchian, volume, ATR)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Get 1w data for trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate Donchian channels (20-period) on 1d
    def calculate_donchian(high_arr, low_arr, window):
        """Donchian Channel: upper = rolling max(high), lower = rolling min(low)"""
        upper = pd.Series(high_arr).rolling(window=window, min_periods=window).max().values
        lower = pd.Series(low_arr).rolling(window=window, min_periods=window).min().values
        return upper, lower
    
    donch_upper, donch_lower = calculate_donchian(high_1d, low_1d, 20)
    
    # Calculate EMA50 on 1w for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume average (20-period) on 1d
    volume_1d_series = pd.Series(volume_1d)
    volume_ma_1d = volume_1d_series.rolling(window=20, min_periods=20).mean().values
    
    # ATR (14-period) for stoploss on 1d
    def calculate_atr(high_arr, low_arr, close_arr, window):
        """Average True Range"""
        tr1 = high_arr - low_arr
        tr2 = np.abs(high_arr - np.roll(close_arr, 1))
        tr3 = np.abs(low_arr - np.roll(close_arr, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]
        atr = pd.Series(tr).ewm(span=window, adjust=False, min_periods=window).mean().values
        return atr
    
    atr_1d = calculate_atr(high_1d, low_1d, close_1d, 14)
    
    # Align all indicators to 1d timeframe
    donch_upper_aligned = align_htf_to_ltf(prices, df_1d, donch_upper)
    donch_lower_aligned = align_htf_to_ltf(prices, df_1d, donch_lower)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    volume_ma_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_1d)
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    entry_price = 0.0
    
    start_idx = 100  # warmup for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donch_upper_aligned[i]) or np.isnan(donch_lower_aligned[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(volume_ma_aligned[i]) or np.isnan(atr_aligned[i])):
            signals[i] = 0.0
            continue
        
        upper = donch_upper_aligned[i]
        lower = donch_lower_aligned[i]
        ema_trend = ema50_1w_aligned[i]
        vol_ma = volume_ma_aligned[i]
        atr_val = atr_aligned[i]
        vol = volume[i]
        price = close[i]
        
        if position == 0:
            # Look for breakouts with volume confirmation and trend alignment
            # Long: price breaks above upper Donchian + volume spike + price > 1w EMA50 (uptrend)
            if price > upper and vol > 1.5 * vol_ma and price > ema_trend:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below lower Donchian + volume spike + price < 1w EMA50 (downtrend)
            elif price < lower and vol > 1.5 * vol_ma and price < ema_trend:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Exit conditions for long
            exit_signal = False
            
            # Exit 1: Price retrace to middle of Donchian channel (mean reversion)
            mid_channel = (upper + lower) / 2
            if price < mid_channel:
                exit_signal = True
            
            # Exit 2: ATR-based stoploss (2.0 * ATR below entry)
            elif price < entry_price - 2.0 * atr_val:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit conditions for short
            exit_signal = False
            
            # Exit 1: Price retrace to middle of Donchian channel (mean reversion)
            mid_channel = (upper + lower) / 2
            if price > mid_channel:
                exit_signal = True
            
            # Exit 2: ATR-based stoploss (2.0 * ATR above entry)
            elif price > entry_price + 2.0 * atr_val:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_1wEMA50_VolumeSpike_ATRStop"
timeframe = "1d"
leverage = 1.0