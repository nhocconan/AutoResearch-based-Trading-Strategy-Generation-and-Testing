#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 12h Williams %R with volume confirmation and choppiness regime filter
# Williams %R identifies overbought/oversold conditions (long when < -80, short when > -20)
# Volume confirmation ensures breakout validity
# Choppiness index regime filter: CHOP > 61.8 = range (mean revert), CHOP < 38.2 = trending (use %R extremes)
# Fixed position size of 0.25 to balance return and drawdown
# Target: 25-60 trades/year on 4h timeframe (100-240 total over 4 years)

name = "4h_12h_williamsr_volume_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 14:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h Williams %R (14-period)
    highest_high_12h = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    lowest_low_12h = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    wr_12h = -100 * (highest_high_12h - close_12h) / (highest_high_12h - lowest_low_12h)
    wr_12h[highest_high_12h == lowest_low_12h] = -50  # Avoid division by zero
    
    # Calculate 12h ATR (14-period) for choppiness index
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr_12h = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_12h[0] = tr1[0]
    atr_12h = pd.Series(tr_12h).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 12h True Range sum (14-period) for choppiness index
    tr_sum_12h = pd.Series(tr_12h).rolling(window=14, min_periods=14).sum().values
    
    # Calculate 12h Choppiness Index (14-period)
    # CHOP = 100 * log10(TR_sum / (ATR * period)) / log10(period)
    atr_period_product = atr_12h * 14
    chop_ratio = np.where(atr_period_product > 0, tr_sum_12h / atr_period_product, 1.0)
    chop_12h = 100 * np.log10(chop_ratio) / np.log10(14)
    
    # Calculate 4h ATR (14-period) for volatility filtering
    tr1_4h = high - low
    tr2_4h = np.abs(high - np.roll(close, 1))
    tr3_4h = np.abs(low - np.roll(close, 1))
    tr_4h = np.maximum(tr1_4h, np.maximum(tr2_4h, tr3_4h))
    tr_4h[0] = tr1_4h[0]
    atr_14_4h = pd.Series(tr_4h).rolling(window=14, min_periods=14).mean().values
    
    # Align all 12h data to 4h timeframe
    wr_12h_aligned = align_htf_to_ltf(prices, df_12h, wr_12h)
    chop_12h_aligned = align_htf_to_ltf(prices, df_12h, chop_12h)
    atr_4h_aligned = align_htf_to_ltf(prices, df_12h, atr_14_4h)  # Use 12h aligned for 4h ATR? No, recalc below
    
    # Recalculate 4h ATR properly (should use 4h data)
    atr_14_4h = pd.Series(tr_4h).rolling(window=14, min_periods=14).mean().values
    
    # Pre-compute volume confirmation (20-period average for 4h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(wr_12h_aligned[i]) or np.isnan(chop_12h_aligned[i]) or
            np.isnan(atr_14_4h[i]) or np.isnan(vol_ma_20[i]) or
            atr_14_4h[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4h volume > 1.5x average 4h volume
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        # Regime filter: use choppiness to determine market state
        chop = chop_12h_aligned[i]
        # In ranging markets (CHOP > 61.8): mean reversion at extremes
        # In trending markets (CHOP < 38.2): continue using %R but with bias
        if chop > 61.8:
            # Ranging market: mean reversion
            long_condition = wr_12h_aligned[i] < -80 and volume_confirmed
            short_condition = wr_12h_aligned[i] > -20 and volume_confirmed
        elif chop < 38.2:
            # Trending market: continue with momentum but extreme %R still signals exhaustion
            long_condition = wr_12h_aligned[i] < -80 and volume_confirmed
            short_condition = wr_12h_aligned[i] > -20 and volume_confirmed
        else:
            # Transition zone: require stronger signals
            long_condition = wr_12h_aligned[i] < -90 and volume_confirmed
            short_condition = wr_12h_aligned[i] > -10 and volume_confirmed
        
        # Fixed position size
        position_size = 0.25
        
        if position == 1:  # Long position
            # Exit when Williams %R returns above -50 (mean reversion) or reaches overbought
            if wr_12h_aligned[i] > -50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
                
        elif position == -1:  # Short position
            # Exit when Williams %R returns below -50 (mean reversion) or reaches oversold
            if wr_12h_aligned[i] < -50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
        else:  # Flat
            if long_condition:
                position = 1
                signals[i] = position_size
            elif short_condition:
                position = -1
                signals[i] = -position_size
    
    return signals