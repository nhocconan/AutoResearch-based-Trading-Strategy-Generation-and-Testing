#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + 1d volume spike + chop regime filter
# - Primary: 12h Williams Alligator (JAW=13, TEETH=8, LIPS=5) for trend direction
# - HTF: 1d volume confirmation (current volume > 2.0x 20-period MA) for conviction
# - Regime: 12h choppy market filter (CHOP(14) > 61.8 = avoid signals in ranging markets)
# - Long: Alligator bullish (LIPS > TEETH > JAW) + volume confirmation + chop < 61.8
# - Short: Alligator bearish (LIPS < TEETH < JAW) + volume confirmation + chop < 61.8
# - Exit: Opposite Alligator alignment or chop > 61.8 (regime change to ranging)
# - Position sizing: 0.25 (discrete level to balance return and drawdown)
# - Works in bull/bear: Alligator adapts to volatility, volume filters false signals, chop regime avoids whipsaws
# - Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe

name = "12h_1d_alligator_volume_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need enough data for calculations
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 25:  # Need enough data for calculations
        return np.zeros(n)
    
    # Pre-compute 12h data
    close_12h = prices['close'].values
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    
    # Pre-compute 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d Williams Alligator (SMMA with periods 13,8,5)
    # Jaw (Blue) - 13-period SMMA
    jaw = np.full(len(close_1d), np.nan)
    # Teeth (Red) - 8-period SMMA
    teeth = np.full(len(close_1d), np.nan)
    # Lips (Green) - 5-period SMMA
    lips = np.full(len(close_1d), np.nan)
    
    # Calculate SMMA (Smoothed Moving Average) - equivalent to EMA with alpha=1/period
    def smma(source, period):
        result = np.full_like(source, np.nan)
        if len(source) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(source[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT_VALUE) / period
        for i in range(period, len(source)):
            if not np.isnan(result[i-1]) and not np.isnan(source[i]):
                result[i] = (result[i-1] * (period-1) + source[i]) / period
        return result
    
    jaw = smma(close_1d, 13)
    teeth = smma(close_1d, 8)
    lips = smma(close_1d, 5)
    
    # Calculate 1d volume moving average (20-period) for volume confirmation
    volume_ma_20_1d = np.full(len(volume_1d), np.nan)
    for i in range(19, len(volume_1d)):
        if not np.isnan(volume_1d[i-19:i+1]).any():
            volume_ma_20_1d[i] = np.mean(volume_1d[i-19:i+1])
    
    # Calculate 12h Choppiness Index (CHOP) - 14 period
    chop = np.full(len(close_12h), np.nan)
    atr_14 = np.full(len(close_12h), np.nan)
    
    # First calculate True Range and ATR(14)
    tr = np.full(len(close_12h), np.nan)
    for i in range(1, len(close_12h)):
        if not (np.isnan(high_12h[i]) or np.isnan(low_12h[i]) or np.isnan(close_12h[i-1])):
            tr[i] = max(
                high_12h[i] - low_12h[i],
                abs(high_12h[i] - close_12h[i-1]),
                abs(low_12h[i] - close_12h[i-1])
            )
    
    # Calculate ATR(14) using Wilder's smoothing (equivalent to EMA with alpha=1/14)
    for i in range(14, len(tr)):
        if not np.isnan(tr[i-13:i+1]).any():
            if i == 14:
                atr_14[i] = np.mean(tr[1:15])  # First ATR is simple average
            else:
                atr_14[i] = (atr_14[i-1] * 13 + tr[i]) / 14
    
    # Calculate Choppiness Index: CHOP = 100 * log10(sum(ATR14)) / log10(14) / log10(max_high - min_low)
    for i in range(27, len(close_12h)):  # Need 14 ATR + 14 period for CHOP
        if not np.isnan(atr_14[i-13:i+1]).any():
            sum_atr = np.sum(atr_14[i-13:i+1])
            if sum_atr > 0:
                max_high = np.max(high_12h[i-13:i+1])
                min_low = np.min(low_12h[i-13:i+1])
                if max_high > min_low:
                    chop[i] = 100 * np.log10(sum_atr) / np.log10(14) / np.log10(max_high - min_low)
    
    # Align all HTF indicators to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    chop_aligned = align_htf_to_ltf(prices, prices, chop)
    volume_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(28, n):  # Start after warmup period for all indicators
        # Skip if any required data is invalid
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(volume_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 2.0x 20-period MA
        volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
        volume_confirm = volume_1d_aligned[i] > 2.0 * volume_ma_20_1d_aligned[i]
        
        # Chop regime filter: only trade when market is trending (CHOP < 61.8)
        trending_regime = chop_aligned[i] < 61.8
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Alligator bullish (Lips > Teeth > Jaw) + volume confirmation + trending regime
            if lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i] and volume_confirm and trending_regime:
                position = 1
                signals[i] = 0.25
            # Short entry: Alligator bearish (Lips < Teeth < Jaw) + volume confirmation + trending regime
            elif lips_aligned[i] < teeth_aligned[i] < jaw_aligned[i] and volume_confirm and trending_regime:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit: Opposite Alligator alignment OR chop > 61.8 (regime change to ranging)
            if position == 1:  # Long position
                if lips_aligned[i] < teeth_aligned[i] or chop_aligned[i] >= 61.8:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                if lips_aligned[i] > teeth_aligned[i] or chop_aligned[i] >= 61.8:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals