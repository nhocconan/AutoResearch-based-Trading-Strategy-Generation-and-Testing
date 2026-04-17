#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout + 1d EMA50 trend filter + volume confirmation + ATR stoploss
- Donchian breakout on 12h provides clear structural entry/exit levels with lower frequency
- 1d EMA50 filters for higher timeframe trend alignment (avoids counter-trend trades)
- Volume spike (>2.0x 20-period average on 12h) confirms breakout strength
- ATR-based stoploss (2.5 * ATR) manages risk
- Target: 12-37 trades/year per symbol (~50-150 total over 4 years)
- Position sizing: 0.25 (discrete levels to minimize fee churn)
- Works in both bull and bear markets: breakouts capture strong moves, trend filter avoids whipsaws in ranging/bear conditions
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for primary calculations (Donchian, volume, ATR)
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Get 1d data for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate Donchian channels (20-period) on 12h
    def calculate_donchian(high_arr, low_arr, window):
        """Donchian Channel: upper = rolling max(high), lower = rolling min(low)"""
        upper = pd.Series(high_arr).rolling(window=window, min_periods=window).max().values
        lower = pd.Series(low_arr).rolling(window=window, min_periods=window).min().values
        return upper, lower
    
    donch_upper, donch_lower = calculate_donchian(high_12h, low_12h, 20)
    
    # Calculate EMA50 on 1d for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume average (20-period) on 12h
    volume_12h_series = pd.Series(volume_12h)
    volume_ma_12h = volume_12h_series.rolling(window=20, min_periods=20).mean().values
    
    # ATR (14-period) for stoploss on 12h
    def calculate_atr(high_arr, low_arr, close_arr, window):
        """Average True Range"""
        tr1 = high_arr - low_arr
        tr2 = np.abs(high_arr - np.roll(close_arr, 1))
        tr3 = np.abs(low_arr - np.roll(close_arr, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]
        atr = pd.Series(tr).ewm(span=window, adjust=False, min_periods=window).mean().values
        return atr
    
    atr_12h = calculate_atr(high_12h, low_12h, close_12h, 14)
    
    # Align all indicators to 12h timeframe (primary timeframe)
    donch_upper_aligned = align_htf_to_ltf(prices, df_12h, donch_upper)
    donch_lower_aligned = align_htf_to_ltf(prices, df_12h, donch_lower)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    volume_ma_aligned = align_htf_to_ltf(prices, df_12h, volume_ma_12h)
    atr_aligned = align_htf_to_ltf(prices, df_12h, atr_12h)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    entry_price = 0.0
    
    start_idx = 100  # warmup for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donch_upper_aligned[i]) or np.isnan(donch_lower_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(volume_ma_aligned[i]) or np.isnan(atr_aligned[i])):
            signals[i] = 0.0
            continue
        
        upper = donch_upper_aligned[i]
        lower = donch_lower_aligned[i]
        ema_trend = ema50_1d_aligned[i]
        vol_ma = volume_ma_aligned[i]
        atr_val = atr_aligned[i]
        vol = volume[i]
        price = close[i]
        
        if position == 0:
            # Look for breakouts with volume confirmation and trend alignment
            # Long: price breaks above upper Donchian + volume spike + price > 1d EMA50 (uptrend)
            if price > upper and vol > 2.0 * vol_ma and price > ema_trend:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below lower Donchian + volume spike + price < 1d EMA50 (downtrend)
            elif price < lower and vol > 2.0 * vol_ma and price < ema_trend:
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
            
            # Exit 2: ATR-based stoploss (2.5 * ATR below entry)
            elif price < entry_price - 2.5 * atr_val:
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
            
            # Exit 2: ATR-based stoploss (2.5 * ATR above entry)
            elif price > entry_price + 2.5 * atr_val:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_1dEMA50_VolumeSpike_ATRStop"
timeframe = "12h"
leverage = 1.0