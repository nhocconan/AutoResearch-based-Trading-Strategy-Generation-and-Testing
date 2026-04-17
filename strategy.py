#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R reversal + 1d EMA50 trend filter + volume confirmation
- Williams %R(14) identifies overbought/oversold conditions for mean reversion entries
- 1d EMA50 ensures alignment with higher timeframe trend to avoid counter-trend trades
- Volume spike (>1.8x 20-period average) confirms momentum behind the reversal
- ATR-based stoploss (2.0 * ATR) manages risk
- Target: 12-30 trades/year per symbol (~50-120 total over 4 years)
- Position sizing: 0.25 (discrete levels to minimize fee churn)
- Works in both bull and bear markets: mean reversion in ranges, trend filter avoids whipsaws in trends
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
    
    # Get 6h data for primary calculations (Williams %R, volume, ATR)
    df_6h = get_htf_data(prices, '6h')
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    volume_6h = df_6h['volume'].values
    
    # Get 1d data for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate Williams %R (14-period) on 6h
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    def calculate_williams_r(high_arr, low_arr, close_arr, window):
        highest_high = pd.Series(high_arr).rolling(window=window, min_periods=window).max().values
        lowest_low = pd.Series(low_arr).rolling(window=window, min_periods=window).min().values
        williams_r = (highest_high - close_arr) / (highest_high - lowest_low) * -100
        # Handle division by zero (when high == low)
        williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
        return williams_r
    
    williams_r = calculate_williams_r(high_6h, low_6h, close_6h, 14)
    
    # Calculate EMA50 on 1d for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume average (20-period) on 6h
    volume_6h_series = pd.Series(volume_6h)
    volume_ma_6h = volume_6h_series.rolling(window=20, min_periods=20).mean().values
    
    # ATR (14-period) for stoploss on 6h
    def calculate_atr(high_arr, low_arr, close_arr, window):
        """Average True Range"""
        tr1 = high_arr - low_arr
        tr2 = np.abs(high_arr - np.roll(close_arr, 1))
        tr3 = np.abs(low_arr - np.roll(close_arr, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]
        atr = pd.Series(tr).ewm(span=window, adjust=False, min_periods=window).mean().values
        return atr
    
    atr_6h = calculate_atr(high_6h, low_6h, close_6h, 14)
    
    # Align all indicators to 6h timeframe (primary timeframe)
    williams_r_aligned = align_htf_to_ltf(prices, df_6h, williams_r)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    volume_ma_aligned = align_htf_to_ltf(prices, df_6h, volume_ma_6h)
    atr_aligned = align_htf_to_ltf(prices, df_6h, atr_6h)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    entry_price = 0.0
    
    start_idx = 100  # warmup for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(volume_ma_aligned[i]) or np.isnan(atr_aligned[i])):
            signals[i] = 0.0
            continue
        
        wr = williams_r_aligned[i]
        ema_trend = ema50_1d_aligned[i]
        vol_ma = volume_ma_aligned[i]
        atr_val = atr_aligned[i]
        vol = volume[i]
        price = close[i]
        
        if position == 0:
            # Look for reversals from extreme Williams %R with volume confirmation and trend alignment
            # Long: Williams %R oversold (< -80) + volume spike + price > 1d EMA50 (uptrend bias)
            if wr < -80 and vol > 1.8 * vol_ma and price > ema_trend:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: Williams %R overbought (> -20) + volume spike + price < 1d EMA50 (downtrend bias)
            elif wr > -20 and vol > 1.8 * vol_ma and price < ema_trend:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Exit conditions for long
            exit_signal = False
            
            # Exit 1: Williams %R returns to neutral territory (> -50)
            if wr > -50:
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
            
            # Exit 1: Williams %R returns to neutral territory (< -50)
            if wr < -50:
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

name = "6h_WilliamsR_1dEMA50_VolumeSpike_ATRStop"
timeframe = "6h"
leverage = 1.0