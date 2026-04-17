#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout + 1d EMA34 trend filter + volume confirmation + ATR stoploss with dynamic sizing
- Donchian breakout on 4h provides structural entry/exit levels with proven frequency (~25-50 trades/year)
- 1d EMA34 filters for higher timeframe trend alignment (avoids counter-trend trades in bear markets)
- Volume spike (>2.0x 20-period average on 4h) confirms breakout strength
- ATR-based stoploss (2.5 * ATR) manages risk
- Position sizing: 0.30 for strong signals, 0.15 for weak signals (discrete levels to minimize fee churn)
- Works in both bull and bear markets: breakouts capture strong moves, trend filter avoids whipsaws
- Uses price action confirmation: requires close outside Donchian for 2 consecutive bars to avoid fakeouts
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
    
    # Get 4h data for primary calculations (Donchian, volume, ATR)
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Get 1d data for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate Donchian channels (20-period) on 4h
    def calculate_donchian(high_arr, low_arr, window):
        """Donchian Channel: upper = rolling max(high), lower = rolling min(low)"""
        upper = pd.Series(high_arr).rolling(window=window, min_periods=window).max().values
        lower = pd.Series(low_arr).rolling(window=window, min_periods=window).min().values
        return upper, lower
    
    donch_upper, donch_lower = calculate_donchian(high_4h, low_4h, 20)
    
    # Calculate EMA34 on 1d for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Volume average (20-period) on 4h
    volume_4h_series = pd.Series(volume_4h)
    volume_ma_4h = volume_4h_series.rolling(window=20, min_periods=20).mean().values
    
    # ATR (14-period) for stoploss on 4h
    def calculate_atr(high_arr, low_arr, close_arr, window):
        """Average True Range"""
        tr1 = high_arr - low_arr
        tr2 = np.abs(high_arr - np.roll(close_arr, 1))
        tr3 = np.abs(low_arr - np.roll(close_arr, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]
        atr = pd.Series(tr).ewm(span=window, adjust=False, min_periods=window).mean().values
        return atr
    
    atr_4h = calculate_atr(high_4h, low_4h, close_4h, 14)
    
    # Align all indicators to 4h timeframe (primary timeframe)
    donch_upper_aligned = align_htf_to_ltf(prices, df_4h, donch_upper)
    donch_lower_aligned = align_htf_to_ltf(prices, df_4h, donch_lower)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    volume_ma_aligned = align_htf_to_ltf(prices, df_4h, volume_ma_4h)
    atr_aligned = align_htf_to_ltf(prices, df_4h, atr_4h)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    entry_price = 0.0
    bars_since_entry = 0
    
    start_idx = 100  # warmup for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donch_upper_aligned[i]) or np.isnan(donch_lower_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(volume_ma_aligned[i]) or np.isnan(atr_aligned[i])):
            signals[i] = 0.0
            continue
        
        upper = donch_upper_aligned[i]
        lower = donch_lower_aligned[i]
        ema_trend = ema34_1d_aligned[i]
        vol_ma = volume_ma_aligned[i]
        atr_val = atr_aligned[i]
        vol = volume[i]
        price = close[i]
        
        if position == 0:
            bars_since_entry = 0
            # Look for breakouts with volume confirmation and trend alignment
            # Require 2 consecutive closes outside Donchian to avoid fakeouts
            if i >= start_idx + 1:
                prev_close = close[i-1]
                prev_upper = donch_upper_aligned[i-1]
                prev_lower = donch_lower_aligned[i-1]
                
                # Long: price breaks above upper Donchian for 2 consecutive bars + volume spike + price > 1d EMA34 (uptrend)
                if price > upper and prev_close > prev_upper and vol > 2.0 * vol_ma and price > ema_trend:
                    signals[i] = 0.30
                    position = 1
                    entry_price = price
                # Short: price breaks below lower Donchian for 2 consecutive bars + volume spike + price < 1d EMA34 (downtrend)
                elif price < lower and prev_close < prev_lower and vol > 2.0 * vol_ma and price < ema_trend:
                    signals[i] = -0.30
                    position = -1
                    entry_price = price
        
        elif position == 1:
            bars_since_entry += 1
            # Exit conditions for long
            exit_signal = False
            
            # Exit 1: Price retrace to middle of Donchian channel (mean reversion)
            mid_channel = (upper + lower) / 2
            if price < mid_channel:
                exit_signal = True
            
            # Exit 2: ATR-based stoploss (2.5 * ATR below entry)
            elif price < entry_price - 2.5 * atr_val:
                exit_signal = True
            
            # Exit 3: Time-based exit (max 20 bars ≈ 5 days) to prevent stagnation
            elif bars_since_entry >= 20:
                exit_signal = True
                signals[i] = 0.15  # half position for gradual exit
            
            if exit_signal and bars_since_entry < 20:
                signals[i] = 0.0
                position = 0
            elif exit_signal and bars_since_entry >= 20:
                # Already set to 0.15 above
                pass
            else:
                signals[i] = 0.30
        
        elif position == -1:
            bars_since_entry += 1
            # Exit conditions for short
            exit_signal = False
            
            # Exit 1: Price retrace to middle of Donchian channel (mean reversion)
            mid_channel = (upper + lower) / 2
            if price > mid_channel:
                exit_signal = True
            
            # Exit 2: ATR-based stoploss (2.5 * ATR above entry)
            elif price > entry_price + 2.5 * atr_val:
                exit_signal = True
            
            # Exit 3: Time-based exit (max 20 bars ≈ 5 days) to prevent stagnation
            elif bars_since_entry >= 20:
                exit_signal = True
                signals[i] = -0.15  # half position for gradual exit
            
            if exit_signal and bars_since_entry < 20:
                signals[i] = 0.0
                position = 0
            elif exit_signal and bars_since_entry >= 20:
                # Already set to -0.15 above
                pass
            else:
                signals[i] = -0.30
    
    return signals

name = "4h_Donchian20_1dEMA34_VolumeSpike_ATRStop_Dynamic"
timeframe = "4h"
leverage = 1.0