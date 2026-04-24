#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator (Jaw/Teeth/Lips) with 1d EMA(34) trend filter and 1d volume spike confirmation.
- Primary timeframe: 12h for entries/exits.
- HTF: 1d EMA(34) for trend direction (bullish if close > EMA34, bearish if close < EMA34).
- Williams Alligator: Jaw (SMA 13, 8-shift), Teeth (SMA 8, 5-shift), Lips (SMA 5, 3-shift).
- Bullish alignment: Lips > Teeth > Jaw. Bearish alignment: Lips < Teeth < Jaw.
- Volume: Current 12h volume > 2.0 * 20-period 1d volume MA to avoid false signals.
- Entry: Long when bullish Alligator alignment AND 1d EMA34 trend bullish AND volume spike.
         Short when bearish Alligator alignment AND 1d EMA34 trend bearish AND volume spike.
- Exit: Opposite Alligator alignment or loss of volume confirmation.
- Signal size: 0.25 discrete to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Williams Alligator on 12h
    # Jaw: SMA(13, 8) - median price smoothed
    median_price = (high + low) / 2.0
    jaw_raw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().shift(8).values
    # Teeth: SMA(8, 5)
    teeth_raw = pd.Series(median_price).rolling(window=8, min_periods=8).mean().shift(5).values
    # Lips: SMA(5, 3)
    lips_raw = pd.Series(median_price).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Get 1d data for EMA(34) trend and volume MA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate EMA(34) on 1d close
    ema_34 = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 20-period volume MA on 1d
    vol_ma_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to 12h
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw_raw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth_raw)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips_raw)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Williams Alligator alignments
    bullish_alignment = (lips_aligned > teeth_aligned) & (teeth_aligned > jaw_aligned)
    bearish_alignment = (lips_aligned < teeth_aligned) & (teeth_aligned < jaw_aligned)
    
    # Volume confirmation: current 12h volume > 2.0 * 20-period 1d volume MA (aligned)
    volume_spike = volume > (2.0 * vol_ma_1d_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20, 13)  # Need enough 1d bars for EMA34, volume MA, and Alligator
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(ema_34_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema_34_val = ema_34_aligned[i]
        curr_close = close[i]
        bull_align = bullish_alignment[i]
        bear_align = bearish_alignment[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Check for entry signals with volume spike
            if vol_spike:
                # Bullish entry: bullish Alligator alignment AND 1d EMA34 bullish (close > EMA34)
                if bull_align and ema_34_val > 0 and curr_close > ema_34_val:
                    signals[i] = 0.25
                    position = 1
                # Bearish entry: bearish Alligator alignment AND 1d EMA34 bearish (close < EMA34)
                elif bear_align and ema_34_val > 0 and curr_close < ema_34_val:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: bearish Alligator alignment OR loss of volume confirmation
            if bear_align or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: bullish Alligator alignment OR loss of volume confirmation
            if bull_align or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsAlligator_1dEMA34Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0