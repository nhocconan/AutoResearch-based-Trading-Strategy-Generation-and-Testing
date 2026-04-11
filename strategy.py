#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator + Elder Ray with weekly volume confirmation and chop regime filter
# - Williams Alligator: Jaw (13,8), Teeth (8,5), Lips (5,3) SMAs on 1d
# - Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13 on 1d
# - Weekly volume confirmation: current 1d volume > 1.5x 20-period weekly average volume (aligned)
# - Choppiness regime filter: CHOP(14) > 61.8 = range (mean revert), CHOP < 38.2 = trending (trend follow)
# - Long: Lips > Teeth > Jaw (bullish alignment) AND Bull Power > 0 AND volume confirmation AND CHOP < 61.8 (not extreme chop)
# - Short: Lips < Teeth < Jaw (bearish alignment) AND Bear Power < 0 AND volume confirmation AND CHOP < 61.8
# - Uses discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 7-25 trades/year (30-100 total over 4 years) to stay within fee drag limits
# - Works in both bull (strong bull power) and bear (strong bear power) markets
# - Chop filter avoids whipsaws in ranging markets

name = "1d_alligator_elder_ray_volume_chop_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load weekly data ONCE before loop for volume confirmation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return signals
    
    # Pre-compute weekly volume confirmation (20-period average)
    volume_1w = df_1w['volume'].values
    volume_sma_20_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_aligned = align_htf_to_ltf(prices, df_1w, volume_sma_20_1w)
    
    # Pre-compute Williams Alligator on 1d timeframe
    # Jaw: 13-period SMA, shifted 8 bars
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().shift(8).values
    # Teeth: 8-period SMA, shifted 5 bars
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().shift(5).values
    # Lips: 5-period SMA, shifted 3 bars
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Pre-compute Elder Ray on 1d timeframe
    # EMA13 for Elder Ray
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    # Bull Power = High - EMA13
    bull_power = high - ema13
    # Bear Power = Low - EMA13
    bear_power = low - ema13
    
    # Pre-compute Choppiness Index on 1d timeframe
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    # Set first TR to 0 (no previous close)
    tr[0] = 0
    # Sum of TR over 14 periods
    tr_sum_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    # Highest high and lowest low over 14 periods
    hh_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    # Choppiness Index: CHOP = 100 * log10(tr_sum_14 / (hh_14 - ll_14)) / log10(14)
    # Avoid division by zero
    hl_range = hh_14 - ll_14
    chop = np.where(hl_range > 0, 100 * np.log10(tr_sum_14 / hl_range) / np.log10(14), 100)
    chop = chop.values  # Ensure it's a numpy array
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(lips[i]) or np.isnan(teeth[i]) or np.isnan(jaw[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(volume_sma_20_aligned[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        volume_current = volume[i]
        
        # Williams Alligator alignment
        bullish_alignment = lips[i] > teeth[i] and teeth[i] > jaw[i]
        bearish_alignment = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        # Elder Ray power
        bull_power_positive = bull_power[i] > 0
        bear_power_negative = bear_power[i] < 0
        
        # Volume confirmation: current volume > 1.5x 20-period weekly average
        vol_confirm = volume_current > 1.5 * volume_sma_20_aligned[i]
        
        # Chop regime filter: avoid extreme chopping markets (CHOP > 61.8)
        chop_filter = chop[i] < 61.8
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: bullish Alligator alignment + positive Bull Power + volume confirmation + chop filter
        if bullish_alignment and bull_power_positive and vol_confirm and chop_filter:
            enter_long = True
        
        # Short: bearish Alligator alignment + negative Bear Power + volume confirmation + chop filter
        if bearish_alignment and bear_power_negative and vol_confirm and chop_filter:
            enter_short = True
        
        # Exit conditions: reverse Alligator alignment or loss of power or extreme chop
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if Alligator turns bearish OR Bull Power becomes negative OR extreme chop
            exit_long = (not bullish_alignment) or (not bull_power_positive) or (chop[i] > 61.8)
        elif position == -1:
            # Exit short if Alligator turns bullish OR Bear Power becomes positive OR extreme chop
            exit_short = (not bearish_alignment) or (not bear_power_negative) or (chop[i] > 61.8)
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals