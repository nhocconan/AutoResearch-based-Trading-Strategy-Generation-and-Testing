#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + 1d volume confirmation + chop regime filter
# - Primary: 12h Williams Alligator (Jaw=TEETH=13, Teeth=TEETH=8, Lips=TEETH=5) for trend direction
# - HTF: 1d volume spike (current volume > 2.0x 50-period MA) for conviction
# - Regime: 1d Choppiness Index (CHOP) > 61.8 = ranging market (fade Alligator signals)
# - Entry: Alligator aligned (Lips > Teeth > Jaw for long, Lips < Teeth < Jaw for short) + volume confirmation + CHOP < 61.8 (trending)
# - Exit: Alligator misalignment OR CHOP > 61.8 (regime shift to ranging)
# - Position sizing: 0.25 (discrete level to balance return and drawdown)
# - Works in bull/bear: Alligator adapts to volatility, volume confirms strength, chop filter avoids whipsaws in ranges
# - Target: 80-120 total trades over 4 years (20-30/year) for 12h timeframe

name = "12h_1d_alligator_volume_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need enough data for Alligator and chop calculations
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need enough data for indicators
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
    
    # Calculate 12h Williams Alligator
    # Jaw: 13-period SMMA of median price, shifted 8 bars
    # Teeth: 8-period SMMA of median price, shifted 5 bars
    # Lips: 5-period SMMA of median price, shifted 3 bars
    median_price_12h = (high_12h + low_12h) / 2.0
    
    def smma(source, period):
        """Smoothed Moving Average"""
        result = np.full_like(source, np.nan)
        if len(source) < period:
            return result
        # First value is SMA
        result[period-1] = np.mean(source[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT_VALUE) / period
        for i in range(period, len(source)):
            if not np.isnan(result[i-1]) and not np.isnan(source[i]):
                result[i] = (result[i-1] * (period-1) + source[i]) / period
        return result
    
    jaw_12h = smma(median_price_12h, 13)
    teeth_12h = smma(median_price_12h, 8)
    lips_12h = smma(median_price_12h, 5)
    
    # Shift as per Alligator definition
    jaw_12h_shifted = np.roll(jaw_12h, 8)
    teeth_12h_shifted = np.roll(teeth_12h, 5)
    lips_12h_shifted = np.roll(lips_12h, 3)
    
    # Set NaN for shifted values that rolled from beginning
    jaw_12h_shifted[:8] = np.nan
    teeth_12h_shifted[:5] = np.nan
    lips_12h_shifted[:3] = np.nan
    
    # Calculate 1d volume moving average (50-period) for volume confirmation
    volume_ma_50_1d = np.full(len(volume_1d), np.nan)
    for i in range(49, len(volume_1d)):
        if not np.isnan(volume_1d[i-49:i+1]).any():
            volume_ma_50_1d[i] = np.mean(volume_1d[i-49:i+1])
    
    # Calculate 1d Choppiness Index (CHOP)
    # CHOP = 100 * log10(TR_sum / (ATR * n)) / log10(n)
    # Where TR_sum = sum of True Range over n periods
    # ATR = average True Range over n periods
    # n = 14 (default)
    chop_period = 14
    
    # Calculate True Range
    tr_1d = np.full(len(high_1d), np.nan)
    tr_1d[0] = high_1d[0] - low_1d[0]  # First bar TR = high - low
    for i in range(1, len(high_1d)):
        if not (np.isnan(high_1d[i]) or np.isnan(low_1d[i]) or np.isnan(close_1d[i-1])):
            tr_1d[i] = max(
                high_1d[i] - low_1d[i],
                abs(high_1d[i] - close_1d[i-1]),
                abs(low_1d[i] - close_1d[i-1])
            )
    
    # Calculate ATR (smoothed TR)
    atr_1d = np.full(len(tr_1d), np.nan)
    if len(tr_1d) >= chop_period:
        # First ATR is average of first chop_period TR values
        atr_1d[chop_period-1] = np.mean(tr_1d[:chop_period])
        # Subsequent ATR values using Wilder's smoothing
        for i in range(chop_period, len(tr_1d)):
            if not np.isnan(atr_1d[i-1]) and not np.isnan(tr_1d[i]):
                atr_1d[i] = (atr_1d[i-1] * (chop_period-1) + tr_1d[i]) / chop_period
    
    # Calculate CHOP
    chop_1d = np.full(len(high_1d), np.nan)
    for i in range(chop_period-1, len(high_1d)):
        if not np.isnan(atr_1d[i]) and atr_1d[i] > 0:
            # Sum of TR over last chop_period periods
            tr_sum = np.sum(tr_1d[i-chop_period+1:i+1])
            # CHOP formula
            chop_1d[i] = 100 * np.log10(tr_sum / (atr_1d[i] * chop_period)) / np.log10(chop_period)
    
    # Align all HTF indicators to 12h timeframe
    jaw_12h_aligned = align_htf_to_ltf(prices, df_1d, jaw_12h_shifted)
    teeth_12h_aligned = align_htf_to_ltf(prices, df_1d, teeth_12h_shifted)
    lips_12h_aligned = align_htf_to_ltf(prices, df_1d, lips_12h_shifted)
    volume_ma_50_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_50_1d)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(1, n):  # Start from second bar to avoid index issues
        # Skip if any required data is invalid
        if (np.isnan(jaw_12h_aligned[i]) or np.isnan(teeth_12h_aligned[i]) or 
            np.isnan(lips_12h_aligned[i]) or np.isnan(volume_ma_50_1d_aligned[i]) or
            np.isnan(chop_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 2.0x 50-period MA
        volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
        volume_confirm = volume_1d_aligned[i] > 2.0 * volume_ma_50_1d_aligned[i]
        
        # Regime filter: CHOP < 61.8 = trending market (good for Alligator)
        trending_regime = chop_1d_aligned[i] < 61.8
        
        if position == 0:  # Flat - look for new entries
            # Alligator aligned for long: Lips > Teeth > Jaw
            long_align = (lips_12h_aligned[i] > teeth_12h_aligned[i] > jaw_12h_aligned[i])
            # Alligator aligned for short: Lips < Teeth < Jaw
            short_align = (lips_12h_aligned[i] < teeth_12h_aligned[i] < jaw_12h_aligned[i])
            
            # Long entry: Alligator long align + volume confirmation + trending regime
            if long_align and volume_confirm and trending_regime:
                position = 1
                signals[i] = 0.25
            # Short entry: Alligator short align + volume confirmation + trending regime
            elif short_align and volume_confirm and trending_regime:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit: Alligator misalignment OR regime shift to ranging (CHOP > 61.8)
            long_align = (lips_12h_aligned[i] > teeth_12h_aligned[i] > jaw_12h_aligned[i])
            short_align = (lips_12h_aligned[i] < teeth_12h_aligned[i] < jaw_12h_aligned[i])
            trending_regime = chop_1d_aligned[i] < 61.8
            
            if position == 1:  # Long position
                if not long_align or not trending_regime:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                if not short_align or not trending_regime:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals