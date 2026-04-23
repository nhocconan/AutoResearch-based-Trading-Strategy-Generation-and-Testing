#!/usr/bin/env python3
"""
Hypothesis: 6h Bollinger Band squeeze breakout with weekly trend filter and volume confirmation.
Long when price breaks above upper Bollinger Band during low volatility (squeeze) AND weekly EMA50 is rising AND volume > 1.5x 20-period average.
Short when price breaks below lower Bollinger Band during low volatility (squeeze) AND weekly EMA50 is falling AND volume > 1.5x 20-period average.
Exit when price retouches Bollinger middle band (20-period SMA) or ATR stoploss hit (2.0*ATR).
Uses discrete position sizing (0.25) to minimize fee churn and control drawdown.
Targets 12-37 trades/year per symbol (50-150 total over 4 years) by requiring Bollinger squeeze (low volatility) before breakout, reducing false signals.
Designed to work in both bull and bear markets by trading with the weekly trend and using volatility-based entry (squeeze breakouts work in all regimes).
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
    open_time = prices['open_time'].values
    
    # Precompute session hours (08-20 UTC) once before loop
    hours = pd.DatetimeIndex(open_time).hour
    
    # Calculate Bollinger Bands (20, 2) on 6h timeframe
    bb_period = 20
    bb_std = 2.0
    sma_bb = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    std_bb = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper_bb = sma_bb + (bb_std * std_bb)
    lower_bb = sma_bb - (bb_std * std_bb)
    middle_bb = sma_bb  # 20-period SMA
    
    # Bollinger Band Width for squeeze detection (low volatility)
    bb_width = (upper_bb - lower_bb) / middle_bb
    bb_width_ma = pd.Series(bb_width).rolling(window=20, min_periods=20).mean().values
    # Squeeze condition: current BB width < 80% of 20-period average BB width
    squeeze_condition = bb_width < (0.8 * bb_width_ma)
    
    # Calculate weekly EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_1w_50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1w_50_aligned = align_htf_to_ltf(prices, df_1w, ema_1w_50)
    
    # EMA slope (rising/falling) - compare current vs 3 periods ago
    ema_slope = np.zeros_like(ema_1w_50_aligned)
    ema_slope[3:] = ema_1w_50_aligned[3:] - ema_1w_50_aligned[:-3]
    
    # Volume average (20-period) on 6h timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for stoploss calculation
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first bar
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from index where all indicators are ready
    start_idx = max(bb_period, 50, 20, 14, 3)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(sma_bb[i]) or np.isnan(std_bb[i]) or np.isnan(upper_bb[i]) or 
            np.isnan(lower_bb[i]) or np.isnan(middle_bb[i]) or np.isnan(squeeze_condition[i]) or
            np.isnan(ema_1w_50_aligned[i]) or np.isnan(ema_slope[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC only
        if hours[i] < 8 or hours[i] > 20:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        upper = upper_bb[i]
        lower = lower_bb[i]
        middle = middle_bb[i]
        squeeze = squeeze_condition[i]
        ema_slope_val = ema_slope[i]
        
        if position == 0:
            # Long: Price breaks above upper Bollinger Band during squeeze AND weekly EMA50 rising AND volume spike
            if (price > upper and 
                squeeze and 
                ema_slope_val > 0 and 
                volume[i] > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: Price breaks below lower Bollinger Band during squeeze AND weekly EMA50 falling AND volume spike
            elif (price < lower and 
                  squeeze and 
                  ema_slope_val < 0 and 
                  volume[i] > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
                entry_price = price
        else:
            # Exit conditions
            exit_signal = False
            
            # Primary exit: Price retouches Bollinger middle band (20-period SMA)
            if position == 1 and price <= middle:
                exit_signal = True
            elif position == -1 and price >= middle:
                exit_signal = True
            
            # ATR-based stoploss: 2.0 * ATR from entry
            if position == 1 and price < entry_price - 2.0 * atr_val:
                exit_signal = True
            elif position == -1 and price > entry_price + 2.0 * atr_val:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_BollingerSqueeze_1wEMA50_Trend_VolumeSpike_ATRStop"
timeframe = "6h"
leverage = 1.0