#!/usr/bin/env python3
"""
Hypothesis: 4h Williams %R + ADX Trend Filter with Volume Spike Confirmation
- Williams %R(14): Measures overbought/oversold levels (-80 to -20 = range, < -80 = oversold, > -20 = overbought)
- ADX > 25: Strong trend - trade in direction of %R extreme (buy oversold in uptrend, sell overbought in downtrend)
- ADX < 20: Weak trend/range - fade extremes (sell overbought, buy oversold)
- Volume Spike: Current volume > 1.8x 20-period average for confirmation
- Uses 1d HTF for trend context (EMA50) to avoid counter-trend trades
- Target: 20-35 trades/year per symbol (~80-140 total over 4 years)
- Position sizing: 0.25 (discrete levels to minimize fee churn)
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
    
    # Get 4h data for primary calculations
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Get 1d data for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate Williams %R on 4h
    def calculate_williams_r(high_arr, low_arr, close_arr, window=14):
        """Williams %R indicator"""
        highest_high = pd.Series(high_arr).rolling(window=window, min_periods=window).max().values
        lowest_low = pd.Series(low_arr).rolling(window=window, min_periods=window).min().values
        wr = -100 * (highest_high - close_arr) / (highest_high - lowest_low)
        # Handle division by zero (when high == low)
        wr[highest_high == lowest_low] = -50
        return wr
    
    wr_4h = calculate_williams_r(high_4h, low_4h, close_4h, 14)
    
    # Calculate ADX on 4h for trend strength
    def calculate_adx(high_arr, low_arr, close_arr, window=14):
        """Average Directional Index"""
        # True Range
        tr1 = high_arr - low_arr
        tr2 = np.abs(high_arr - np.roll(close_arr, 1))
        tr3 = np.abs(low_arr - np.roll(close_arr, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]
        
        # Directional Movement
        dm_plus = np.where((high_arr - np.roll(high_arr, 1)) > (np.roll(low_arr, 1) - low_arr),
                           np.maximum(high_arr - np.roll(high_arr, 1), 0), 0)
        dm_minus = np.where((np.roll(low_arr, 1) - low_arr) > (high_arr - np.roll(high_arr, 1)),
                            np.maximum(np.roll(low_arr, 1) - low_arr, 0), 0)
        dm_plus[0] = 0
        dm_minus[0] = 0
        
        # Smoothed values
        tr_smoothed = pd.Series(tr).ewm(span=window, adjust=False, min_periods=window).mean().values
        dm_plus_smoothed = pd.Series(dm_plus).ewm(span=window, adjust=False, min_periods=window).mean().values
        dm_minus_smoothed = pd.Series(dm_minus).ewm(span=window, adjust=False, min_periods=window).mean().values
        
        # Directional Indicators
        di_plus = 100 * dm_plus_smoothed / tr_smoothed
        di_minus = 100 * dm_minus_smoothed / tr_smoothed
        
        # DX and ADX
        dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
        dx[np.isnan(dx)] = 0
        adx = pd.Series(dx).ewm(span=window, adjust=False, min_periods=window).mean().values
        return adx
    
    adx_4h = calculate_adx(high_4h, low_4h, close_4h, 14)
    
    # Calculate EMA50 on 1d for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
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
    
    # Align all indicators to 4h timeframe
    wr_aligned = align_htf_to_ltf(prices, df_4h, wr_4h)
    adx_aligned = align_htf_to_ltf(prices, df_4h, adx_4h)
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    volume_ma_aligned = align_htf_to_ltf(prices, df_4h, volume_ma_4h)
    atr_aligned = align_htf_to_ltf(prices, df_4h, atr_4h)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    entry_price = 0.0
    
    start_idx = 100  # warmup for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(wr_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(ema50_aligned[i]) or np.isnan(volume_ma_aligned[i]) or np.isnan(atr_aligned[i])):
            signals[i] = 0.0
            continue
        
        wr = wr_aligned[i]
        adx = adx_aligned[i]
        ema50 = ema50_aligned[i]
        vol_ma = volume_ma_aligned[i]
        atr_val = atr_aligned[i]
        vol = volume[i]
        price = close[i]
        
        if position == 0:
            # Entry logic based on trend regime
            if adx > 25:  # Strong trend
                # Determine trend direction from 1d EMA50
                uptrend = price > ema50
                # Long: Oversold in uptrend + volume spike
                if wr < -80 and uptrend and vol > 1.8 * vol_ma:
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
                # Short: Overbought in downtrend + volume spike
                elif wr > -20 and not uptrend and vol > 1.8 * vol_ma:
                    signals[i] = -0.25
                    position = -1
                    entry_price = price
            else:  # Weak trend/range (ADX < 25)
                # Fade extremes: sell overbought, buy oversold
                if wr > -20 and vol > 1.8 * vol_ma:  # Overbought
                    signals[i] = -0.25
                    position = -1
                    entry_price = price
                elif wr < -80 and vol > 1.8 * vol_ma:  # Oversold
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
        
        elif position == 1:
            # Exit conditions for long
            exit_signal = False
            
            # Exit 1: Williams %R returns to neutral territory
            if wr > -50:  # Returned from oversold
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
            
            # Exit 1: Williams %R returns to neutral territory
            if wr < -50:  # Returned from overbought
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

name = "4h_WilliamsR_ADXTrend_VolumeSpike_ATRStop"
timeframe = "4h"
leverage = 1.0