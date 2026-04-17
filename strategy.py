#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla pivot breakout with 1d EMA50 trend filter and volume confirmation
- Uses 1d Camarilla levels (H3, L3, H4, L4) calculated from prior 1d candle
- Breakout above H3 or below L3 with volume > 1.5x 20-period average
- Trend filter: price must be above/below 1d EMA50 for long/short respectively
- Exit: price retracement to H3/L3 level or ATR-based stop (2.0 * ATR)
- Position sizing: 0.25 (discrete to minimize fee churn)
- Target: 12-30 trades/year per symbol (~50-120 total over 4 years)
- Works in both bull and bear markets: breakouts capture strong moves, trend filter avoids counter-trend trades
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
    
    # Get 1d data for Camarilla pivots and EMA50 (HTF)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Camarilla levels from prior 1d candle
    # H4 = close + 1.5 * (high - low)
    # H3 = close + 1.1 * (high - low)
    # L3 = close - 1.1 * (high - low)
    # L4 = close - 1.5 * (high - low)
    daily_range = high_1d - low_1d
    H4 = close_1d + 1.5 * daily_range
    H3 = close_1d + 1.1 * daily_range
    L3 = close_1d - 1.1 * daily_range
    L4 = close_1d - 1.5 * daily_range
    
    # Volume average (20-period) on 1d
    volume_1d = df_1d['volume'].values
    volume_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
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
    
    # Align all 1d indicators to 6h timeframe (primary timeframe)
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    volume_ma_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_1d)
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    entry_price = 0.0
    
    start_idx = 100  # warmup for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(volume_ma_aligned[i]) or np.isnan(atr_aligned[i])):
            signals[i] = 0.0
            continue
        
        H3 = H3_aligned[i]
        L3 = L3_aligned[i]
        H4 = H4_aligned[i]
        L4 = L4_aligned[i]
        ema_trend = ema50_1d_aligned[i]
        vol_ma = volume_ma_aligned[i]
        atr_val = atr_aligned[i]
        vol = volume[i]
        price = close[i]
        
        if position == 0:
            # Look for breakouts with volume confirmation and trend alignment
            # Long: price breaks above H3 + volume spike + price > 1d EMA50 (uptrend)
            if price > H3 and vol > 1.5 * vol_ma and price > ema_trend:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below L3 + volume spike + price < 1d EMA50 (downtrend)
            elif price < L3 and vol > 1.5 * vol_ma and price < ema_trend:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Exit conditions for long
            exit_signal = False
            
            # Exit 1: Price retrace to H3 level (take profit)
            if price < H3:
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
            
            # Exit 1: Price retrace to L3 level (take profit)
            if price > L3:
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

name = "6h_Camarilla_H3L3_1dEMA50_VolumeSpike_ATRStop"
timeframe = "6h"
leverage = 1.0