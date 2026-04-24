#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1w EMA34 trend filter and volume spike confirmation.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1w for EMA34 trend filter (more stable than daily for long-term bias).
- Elder Ray: Bull Power = High - EMA13(Close), Bear Power = Low - EMA13(Close).
- Entry: Long when Bull Power > 0 AND price > 1w EMA34 AND volume > 2.0 * 6h volume MA(20);
         Short when Bear Power < 0 AND price < 1w EMA34 AND volume > 2.0 * 6h volume MA(20).
- Exit: Close-based reversal (opposite signal) or stoploss via trend filter (signal=0 when price closes below/above 1w EMA34).
- Signal size: 0.25 discrete to minimize fee drag while maintaining profit potential.
- Elder Ray measures bull/bear strength relative to short-term trend (EMA13); 1w EMA34 filters for long-term trend alignment.
- Works in bull markets via trend-aligned strength and bear markets via weakness filtering.
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
    
    # Calculate EMA13 for Elder Ray (short-term trend)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema_13  # Bull Power: High - EMA13
    bear_power = low - ema_13   # Bear Power: Low - EMA13
    
    # Get 1w data for EMA34 (long-term trend filter)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate EMA(34) on 1w
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Get 6h data for volume MA
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    volume_6h = df_6h['volume'].values
    
    # Calculate volume MA(20) on 6h
    vol_ma_6h = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to primary 6h timeframe
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    vol_ma_aligned = align_htf_to_ltf(prices, df_6h, vol_ma_6h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 34  # EMA34 needs 34 periods (1w data)
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_ma_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        curr_bull = bull_power[i]
        curr_bear = bear_power[i]
        
        # Stoploss: exit if price closes below/above 1w EMA34 (trend filter)
        if position == 1:
            if curr_close < ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:
            if curr_close > ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
        
        # Entry conditions with volume confirmation and trend filter
        bullish_entry = curr_bull > 0  # Bull Power positive
        bearish_entry = curr_bear < 0  # Bear Power negative
        
        # Trend filter from 1w EMA34
        price_above_ema = curr_close > ema_34_1w_aligned[i]
        price_below_ema = curr_close < ema_34_1w_aligned[i]
        
        # Volume confirmation
        vol_confirm = curr_volume > 2.0 * vol_ma_aligned[i]
        
        if position == 0:
            # Check for entry signals
            if vol_confirm:
                # Long: Bull Power > 0 AND price above 1w EMA34
                if bullish_entry and price_above_ema:
                    signals[i] = 0.25
                    position = 1
                # Short: Bear Power < 0 AND price below 1w EMA34
                elif bearish_entry and price_below_ema:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_BullBearPower_1wEMA34_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0