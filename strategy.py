#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla H3/L3 breakout with 1d EMA34 trend filter and volume spike confirmation.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 1d EMA34 for trend filter (price > EMA34 = uptrend, price < EMA34 = downtrend).
- Entry: Long when price breaks above H3 level AND volume > 2.0 * 4h volume MA(20);
         Short when price breaks below L3 level AND volume > 2.0 * 4h volume MA(20).
- Exit: Close below/above L3/H3 levels respectively for profit-taking, with ATR-based stoploss (2.0 * ATR(14)).
- Signal size: 0.25 discrete to control fee drag.
- Uses Camarilla pivot levels for intraday support/resistance, volume confirmation for participation,
  1d EMA34 trend filter to avoid counter-trend trades, and ATR for risk management.
- Designed to work in both bull and bear markets via trend filter and breakout logic.
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
    
    # Get 4h data for Camarilla calculation, volume MA(20), and ATR(14)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Calculate ATR(14) for 4h timeframe
    tr1 = high_4h[1:] - low_4h[1:]
    tr2 = np.abs(high_4h[1:] - close_4h[:-1])
    tr3 = np.abs(low_4h[1:] - close_4h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[high_4h[0] - low_4h[0]], tr])  # first TR is high-low
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate volume MA(20) for 4h timeframe
    vol_ma_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Camarilla pivot levels for 4h timeframe
    # Based on previous day's high, low, close
    # H3 = close + 1.1 * (high - low) / 4
    # L3 = close - 1.1 * (high - low) / 4
    # We need to use the previous 4h bar's high/low/close for current bar's levels
    prev_high = np.concatenate([[high_4h[0]], high_4h[:-1]])  # shift by 1
    prev_low = np.concatenate([[low_4h[0]], low_4h[:-1]])
    prev_close = np.concatenate([[close_4h[0]], close_4h[:-1]])
    
    camarilla_h3 = prev_close + 1.1 * (prev_high - prev_low) / 4
    camarilla_l3 = prev_close - 1.1 * (prev_high - prev_low) / 4
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA34 to 4h timeframe
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20, 14)  # EMA34 needs 34, volume MA needs 20, ATR needs 14
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_aligned[i]) or 
            np.isnan(camarilla_h3[i]) or 
            np.isnan(camarilla_l3[i]) or 
            np.isnan(vol_ma_4h[i]) or 
            np.isnan(atr14[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_atr = atr14[i]
        
        # Volume confirmation: 2.0x threshold for strict entry
        vol_confirm = curr_volume > 2.0 * vol_ma_4h[i]
        
        if position == 0:
            # Check for entry signals
            if vol_confirm:
                # Long: price breaks above H3 level AND price > 1d EMA34 (uptrend)
                if curr_high > camarilla_h3[i] and curr_close > ema_34_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Short: price breaks below L3 level AND price < 1d EMA34 (downtrend)
                elif curr_low < camarilla_l3[i] and curr_close < ema_34_aligned[i]:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        elif position == 1:
            # Long position: check exit conditions
            # Stoploss: 2.0 * ATR below entry
            stoploss = entry_price - 2.0 * curr_atr
            # Profit take: close below L3 level (reversal signal)
            if curr_close < stoploss or curr_close < camarilla_l3[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: check exit conditions
            # Stoploss: 2.0 * ATR above entry
            stoploss = entry_price + 2.0 * curr_atr
            # Profit take: close above H3 level (reversal signal)
            if curr_close > stoploss or curr_close > camarilla_h3[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_H3L3_Breakout_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0