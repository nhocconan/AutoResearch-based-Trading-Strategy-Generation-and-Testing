#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla H3/L3 breakout with 1d EMA34 trend filter and volume confirmation.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 1d EMA34 for trend filter (defines bull/bear regime).
- Entry: Long when price breaks above Camarilla H3 level in bull regime with volume > 1.5 * 4h volume MA(20);
         Short when price breaks below Camarilla L3 level in bear regime with volume > 1.5 * 4h volume MA(20).
- Exit: Opposite Camarilla breakout (L3 for long, H3 for short) or ATR trailing stop (3.0 * ATR(14)).
- Signal size: 0.25 discrete to balance capture and fee control.
- Camarilla levels provide intraday support/resistance; EMA34 adapts to trend; volume confirms conviction.
- Works in bull (breakouts with trend) and bear (strong moves after regime shifts).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Camarilla calculation and volume
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Get 1d data for EMA34 calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    close_1d = df_1d['close']
    ema_34 = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align EMA34 to 4h timeframe
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Calculate 4h Camarilla levels (H3, L3, H4, L4)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Camarilla formula: based on previous day's range
    # H4 = close + 1.1 * (high - low) / 2
    # H3 = close + 1.1 * (high - low) / 4
    # L3 = close - 1.1 * (high - low) / 4
    # L4 = close - 1.1 * (high - low) / 2
    camarilla_high = (close_4h + 1.1 * (high_4h - low_4h) / 4).values  # H3
    camarilla_low = (close_4h - 1.1 * (high_4h - low_4h) / 4).values   # L3
    camarilla_high_aligned = align_htf_to_ltf(prices, df_4h, camarilla_high)
    camarilla_low_aligned = align_htf_to_ltf(prices, df_4h, camarilla_low)
    
    # Calculate 4h volume MA(20) for confirmation
    volume_4h = df_4h['volume'].values
    vol_ma_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    vol_ma_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_4h)
    
    # Calculate 4h ATR(14) for trailing stop
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0
    lowest_since_entry = 0
    
    # Start from index where all indicators are ready
    start_idx = max(35, 30, 20, 14, 1)  # EMA34 needs 35, Camarilla needs 30, volume MA needs 20, ATR needs 14
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_aligned[i]) or 
            np.isnan(camarilla_high_aligned[i]) or np.isnan(camarilla_low_aligned[i]) or 
            np.isnan(vol_ma_4h_aligned[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Volume confirmation: 1.5x threshold (balanced to reduce trades)
        vol_confirm = curr_volume > 1.5 * vol_ma_4h_aligned[i]
        
        # Trend filter: EMA34 direction
        bull_regime = close[i] > ema_34_aligned[i]
        bear_regime = close[i] < ema_34_aligned[i]
        
        if position == 0:
            # Check for entry signals
            # Long: price breaks above Camarilla H3 in bull regime with volume confirmation
            if curr_close > camarilla_high_aligned[i] and bull_regime and vol_confirm:
                signals[i] = 0.25
                position = 1
                highest_since_entry = curr_high
            # Short: price breaks below Camarilla L3 in bear regime with volume confirmation
            elif curr_close < camarilla_low_aligned[i] and bear_regime and vol_confirm:
                signals[i] = -0.25
                position = -1
                lowest_since_entry = curr_low
        elif position == 1:
            # Long position: update highest and check exit conditions
            highest_since_entry = max(highest_since_entry, curr_high)
            # Exit: ATR trailing stop or opposite breakout (below Camarilla L3)
            if curr_low <= highest_since_entry - 3.0 * atr[i] or curr_close < camarilla_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: update lowest and check exit conditions
            lowest_since_entry = min(lowest_since_entry, curr_low)
            # Exit: ATR trailing stop or opposite breakout (above Camarilla H3)
            if curr_high >= lowest_since_entry + 3.0 * atr[i] or curr_close > camarilla_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_H3L3_1dEMA34_Trend_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0