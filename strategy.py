#!/usr/bin/env python3
"""
Hypothesis: 4h Williams Alligator with 1d EMA34 trend filter and volume confirmation.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 1d EMA34 for trend filter (price above/below EMA34 defines bull/bear regime).
- Williams Alligator: Jaw (13-period SMMA, 8-bar shift), Teeth (8-period SMMA, 5-bar shift), Lips (5-period SMMA, 3-bar shift).
- Entry: Long when Lips > Teeth > Jaw (bullish alignment) in bull regime with volume > 1.5 * 4h volume MA(20);
         Short when Lips < Teeth < Jaw (bearish alignment) in bear regime with volume > 1.5 * 4h volume MA(20).
- Exit: ATR trailing stop (2.0 * ATR(14)) or opposite Alligator alignment.
- Signal size: 0.25 discrete to balance capture and fee control.
- Designed for BTC/ETH: Alligator identifies trend phases, EMA34 filter avoids chop, volume confirms participation.
  Works in bull (trend continuation) and bear (strong moves after reversals).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(values, period):
    """Smoothed Moving Average (SMMA) - same as RMA/Wilder's"""
    if len(values) < period:
        return np.full_like(values, np.nan, dtype=float)
    result = np.full_like(values, np.nan, dtype=float)
    # First value is SMA
    result[period-1] = np.mean(values[:period])
    # Subsequent values: SMMA = (prev_smma * (period-1) + current_value) / period
    for i in range(period, len(values)):
        result[i] = (result[i-1] * (period-1) + values[i]) / period
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Alligator and volume
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
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
    
    # Calculate Williams Alligator on 4h median price
    median_price = (high + low) / 2
    
    # Jaw: 13-period SMMA, 8 bars ahead
    jaw = smma(median_price, 13)
    jaw = np.roll(jaw, 8)
    # Teeth: 8-period SMMA, 5 bars ahead
    teeth = smma(median_price, 8)
    teeth = np.roll(teeth, 5)
    # Lips: 5-period SMMA, 3 bars ahead
    lips = smma(median_price, 5)
    lips = np.roll(lips, 3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0
    lowest_since_entry = 0
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20, 14, 13+8, 8+5, 5+3)  # EMA34 needs 34, volume MA needs 20, ATR needs 14, Alligator shifts
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_4h_aligned[i]) or 
            np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Volume spike confirmation: 1.5x threshold (balanced to reduce trades)
        vol_spike = curr_volume > 1.5 * vol_ma_4h_aligned[i]
        
        # Trend filter: price above/below 1d EMA34
        bull_regime = curr_close > ema_34_1d_aligned[i]
        bear_regime = curr_close < ema_34_1d_aligned[i]
        
        # Alligator alignment
        bullish_alignment = lips[i] > teeth[i] and teeth[i] > jaw[i]
        bearish_alignment = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        if position == 0:
            # Check for entry signals
            # Long: bullish Alligator alignment in bull regime with volume spike
            if bullish_alignment and bull_regime and vol_spike:
                signals[i] = 0.25
                position = 1
                highest_since_entry = curr_high
            # Short: bearish Alligator alignment in bear regime with volume spike
            elif bearish_alignment and bear_regime and vol_spike:
                signals[i] = -0.25
                position = -1
                lowest_since_entry = curr_low
        elif position == 1:
            # Long position: update highest and check exit conditions
            highest_since_entry = max(highest_since_entry, curr_high)
            # Exit: ATR trailing stop or opposite signal (bearish alignment)
            if curr_low <= highest_since_entry - 2.0 * atr[i] or bearish_alignment:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: update lowest and check exit conditions
            lowest_since_entry = min(lowest_since_entry, curr_low)
            # Exit: ATR trailing stop or opposite signal (bullish alignment)
            if curr_high >= lowest_since_entry + 2.0 * atr[i] or bullish_alignment:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WilliamsAlligator_1dEMA34_Trend_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0