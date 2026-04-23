#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R Reversal + 12h EMA Trend + Volume Spike + Choppiness Regime Filter.
Long when Williams %R crosses above -80 (oversold reversal) AND close > 12h EMA50 AND volume > 2.0x 20-period average AND chop < 61.8 (trending).
Short when Williams %R crosses below -20 (overbought reversal) AND close < 12h EMA50 AND volume > 2.0x 20-period average AND chop < 61.8.
Exit when Williams %R crosses above -20 (for longs) or below -80 (for shorts) or ATR stoploss (2.5x ATR).
Uses discrete position sizing (0.25) to minimize fee churn. Targets 12-30 trades/year per symbol.
Adds 12h trend filter and choppiness regime to avoid ranging markets and improve BTC/ETH performance.
Williams %R is a momentum oscillator that identifies overbought/oversold conditions, effective in both bull and bear markets when combined with trend and volume filters.
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
    
    # Calculate Williams %R on 6h data (period=14)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50.0, williams_r)  # neutral when no range
    
    # Volume average (20-period) on 6h timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Load 12h data for EMA50 and choppiness filter - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate EMA50 on 12h data
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Choppiness Index on 12h data (period=14)
    tr_12h = []
    for i in range(len(high_12h)):
        tr = max(high_12h[i] - low_12h[i], 
                abs(high_12h[i] - close_12h[i-1]) if i > 0 else 0, 
                abs(low_12h[i] - close_12h[i-1]) if i > 0 else 0)
        tr_12h.append(tr)
    atr_12h = pd.Series(tr_12h).rolling(window=14, min_periods=14).mean().values
    
    max_high_12h = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    min_low_12h = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    range_sum_12h = atr_12h * 14
    max_min_diff_12h = max_high_12h - min_low_12h
    chop_12h = np.where(
        (range_sum_12h > 0) & (max_min_diff_12h > 0),
        100 * np.log10(range_sum_12h / max_min_diff_12h) / np.log10(14),
        50.0  # neutral when undefined
    )
    
    # Calculate ATR on 6h data for stoploss
    tr1 = np.maximum(high - low, np.abs(high - np.roll(close, 1)))
    tr2 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, tr2)
    tr[0] = high[0] - low[0]  # first bar
    atr_6h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align 12h indicators to 6h timeframe
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    chop_12h_aligned = align_htf_to_ltf(prices, df_12h, chop_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(williams_r[i]) or np.isnan(ema50_12h_aligned[i]) or 
            np.isnan(chop_12h_aligned[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(atr_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # Long: Williams %R crosses above -80 (oversold reversal) AND close > 12h EMA50 AND volume spike AND trending market (chop < 61.8)
            if (williams_r[i] > -80 and williams_r[i-1] <= -80 and 
                close[i] > ema50_12h_aligned[i] and 
                volume[i] > 2.0 * vol_ma_val and
                chop_12h_aligned[i] < 61.8):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: Williams %R crosses below -20 (overbought reversal) AND close < 12h EMA50 AND volume spike AND trending market (chop < 61.8)
            elif (williams_r[i] < -20 and williams_r[i-1] >= -20 and 
                  close[i] < ema50_12h_aligned[i] and 
                  volume[i] > 2.0 * vol_ma_val and
                  chop_12h_aligned[i] < 61.8):
                signals[i] = -0.25
                position = -1
                entry_price = price
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Williams %R crosses above -20 (overbought) or ATR stoploss
                if williams_r[i] >= -20 and williams_r[i-1] < -20:
                    exit_signal = True
                elif price < entry_price - 2.5 * atr_6h[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Williams %R crosses below -80 (oversold) or ATR stoploss
                if williams_r[i] <= -80 and williams_r[i-1] > -80:
                    exit_signal = True
                elif price > entry_price + 2.5 * atr_6h[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_WilliamsR_Reversal_12hEMA50_VolumeSpike_ChopFilter"
timeframe = "6h"
leverage = 1.0