#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d volume confirmation and choppiness regime filter
# Williams Alligator (Jaw=TEETH=LIPS) identifies trending vs ranging markets
# In trending markets (JAW > TEETH > LIPS for long, reverse for short), we trade breakouts
# In ranging markets (Alligator lines intertwined), we fade extremes at Bollinger Bands
# 1d volume spike confirms institutional participation
# Designed for 12h timeframe targeting 12-37 trades/year (50-150 total over 4 years)
# Uses discrete position sizing (0.25) to minimize fee churn and control drawdown
# Works in bull markets (trend + breakout) and bear markets (mean reversion in range)

name = "12h_WilliamsAlligator_1dVolume_Chop_Regime"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for volume confirmation and choppiness regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Williams Alligator on 12h close prices (Smoothed Moving Average - SMA equivalent)
    # Jaw: 13-period SMMA, shifted 8 bars ahead
    # Teeth: 8-period SMMA, shifted 5 bars ahead  
    # Lips: 5-period SMMA, shifted 3 bars ahead
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().shift(8)
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().shift(5)
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().shift(3)
    
    jaw_values = jaw.values
    teeth_values = teeth.values
    lips_values = lips.values
    
    # 1d volume confirmation (20-period EMA)
    vol_ema_20_1d = pd.Series(df_1d['volume']).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_confirmation_1d = df_1d['volume'].values > (1.5 * vol_ema_20_1d)
    volume_confirmation_aligned = align_htf_to_ltf(prices, df_1d, volume_confirmation_1d)
    
    # 1d choppiness regime (CHOP > 61.8 = range, CHOP < 38.2 = trend)
    # True Range
    tr1 = pd.Series(df_1d['high']).diff().abs()
    tr2 = pd.Series(df_1d['low']).diff().abs()
    tr3 = (pd.Series(df_1d['close']).shift() - pd.Series(df_1d['high'])).abs()
    tr4 = (pd.Series(df_1d['close']).shift() - pd.Series(df_1d['low'])).abs()
    tr = pd.concat([tr1, tr2, tr3, tr4], axis=1).max(axis=1)
    atr_1d = tr.rolling(window=14, min_periods=14).sum()
    
    # Chop = 100 * log10( sum(ATR14) / (max(high)-min(low)) * 1/sqrt(14) )
    max_high = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max()
    min_low = pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min()
    chop = 100 * np.log10(atr_1d / (max_high - min_low) * (1 / np.sqrt(14)))
    chop_values = chop.values
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for Alligator and indicators)
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(jaw_values[i]) or np.isnan(teeth_values[i]) or np.isnan(lips_values[i]) or
            np.isnan(volume_confirmation_aligned[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Determine market regime from 1d choppiness
        is_ranging = chop_aligned[i] > 61.8
        is_trending = chop_aligned[i] < 38.2
        
        if position == 0:  # Flat - look for new entries
            if is_trending:
                # Trending market: Alligator breakout
                # Long: Lips > Teeth > Jaw (bullish alignment) AND price > Lips
                # Short: Lips < Teeth < Jaw (bearish alignment) AND price < Lips
                if lips_values[i] > teeth_values[i] > jaw_values[i] and close[i] > lips_values[i]:
                    signals[i] = 0.25
                    position = 1
                elif lips_values[i] < teeth_values[i] < jaw_values[i] and close[i] < lips_values[i]:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            elif is_ranging:
                # Ranging market: Mean reversion at Bollinger Bands (20, 2.0)
                # Calculate Bollinger Bands on 12h close
                sma_20 = pd.Series(close[:i+1]).rolling(window=20, min_periods=20).mean().iloc[-1]
                std_20 = pd.Series(close[:i+1]).rolling(window=20, min_periods=20).std().iloc[-1]
                upper_band = sma_20 + (2.0 * std_20)
                lower_band = sma_20 - (2.0 * std_20)
                
                # Long at lower band with volume confirmation
                # Short at upper band with volume confirmation
                if close[i] <= lower_band and volume_confirmation_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                elif close[i] >= upper_band and volume_confirmation_aligned[i]:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                # Transition zone - no trade
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit conditions
            if is_trending:
                # Exit trend long: Lips < Teeth (Alligator sleeping) OR price < Teeth
                if lips_values[i] < teeth_values[i] or close[i] < teeth_values[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif is_ranging:
                # Exit range long: price > SMA(20) OR volume confirmation lost
                sma_20 = pd.Series(close[:i+1]).rolling(window=20, min_periods=20).mean().iloc[-1]
                if close[i] >= sma_20 or not volume_confirmation_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:  # Short position
            # Exit conditions
            if is_trending:
                # Exit trend short: Lips > Teeth (Alligator sleeping) OR price > Teeth
                if lips_values[i] > teeth_values[i] or close[i] > teeth_values[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            elif is_ranging:
                # Exit range short: price < SMA(20) OR volume confirmation lost
                sma_20 = pd.Series(close[:i+1]).rolling(window=20, min_periods=20).mean().iloc[-1]
                if close[i] <= sma_20 or not volume_confirmation_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:
                signals[i] = 0.0
                position = 0
    
    return signals