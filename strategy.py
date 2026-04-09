#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 1d/1w regime + volume confirmation
# - Primary signal: Elder Ray (Bull Power = EMA13 - Low, Bear Power = High - EMA13) on 6h
# - Long when Bull Power > 0 AND Bear Power < previous Bear Power (bullish momentum building)
# - Short when Bear Power < 0 AND Bull Power < previous Bull Power (bearish momentum building)
# - Regime filter: 1d price > 200 EMA for long bias, < 200 EMA for short bias
# - Weekly filter: 1w close > 1w open for additional long bias, < for short bias
# - Volume confirmation: 6h volume > 20-period median volume
# - Position size: 0.25 (discrete level) to minimize fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) per 6h strategy guidelines
# - Works in bull/bear: Elder Ray measures power, higher timeframe filters ensure alignment with dominant trend

name = "6h_1d_1w_elderray_regime_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 200 or len(df_1w) < 50:
        return np.zeros(n)
    
    # Pre-compute 1d EMA200 for trend filter
    close_1d = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Pre-compute 1w bullish/bearish bias (close > open = bullish)
    open_1w = df_1w['open'].values
    close_1w = df_1w['close'].values
    weekly_bullish = close_1w > open_1w  # True for bullish weekly candle
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish.astype(float))
    
    # Pre-compute Elder Ray on 6h timeframe
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 30:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # EMA13 for Elder Ray
    ema_13_6h = pd.Series(close_6h).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power_6h = ema_13_6h - low_6h      # Bull Power = EMA13 - Low
    bear_power_6h = high_6h - ema_13_6h     # Bear Power = High - EMA13
    
    # Align Elder Ray components to primary timeframe (completed 6h bar only)
    bull_power_aligned = align_htf_to_ltf(prices, df_6h, bull_power_6h)
    bear_power_aligned = align_htf_to_ltf(prices, df_6h, bear_power_6h)
    
    # 6h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 6h volume regime: volume > 20-period median volume
    median_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    volume_regime = volume > median_volume_20
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_200_aligned[i]) or
            np.isnan(weekly_bullish_aligned[i]) or
            np.isnan(bull_power_aligned[i]) or
            np.isnan(bear_power_aligned[i]) or
            np.isnan(volume_regime[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Bear Power turns positive OR weekly bias turns bearish OR volume dries up
            if (bear_power_aligned[i] > 0 or 
                weekly_bullish_aligned[i] < 0.5 or 
                not volume_regime[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Bull Power turns positive OR weekly bias turns bullish OR volume dries up
            if (bull_power_aligned[i] > 0 or 
                weekly_bullish_aligned[i] > 0.5 or 
                not volume_regime[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Elder Ray alignment with volume confirmation and regime filters
            # Long: Bull Power > 0 (bullish energy) AND Bear Power declining (weakening bears) 
            #       AND 1d price > EMA200 (bullish trend) AND weekly bullish AND volume regime
            if (bull_power_aligned[i] > 0 and 
                bear_power_aligned[i] < bear_power_aligned[max(0, i-1)] and  # Bear Power decreasing
                close[i] > ema_200_aligned[i] and 
                weekly_bullish_aligned[i] > 0.5 and 
                volume_regime[i]):
                position = 1
                signals[i] = 0.25
            # Short: Bear Power > 0 (bearish energy) AND Bull Power declining (weakening bulls)
            #        AND 1d price < EMA200 (bearish trend) AND weekly bearish AND volume regime
            elif (bear_power_aligned[i] > 0 and 
                  bull_power_aligned[i] < bull_power_aligned[max(0, i-1)] and  # Bull Power decreasing
                  close[i] < ema_200_aligned[i] and 
                  weekly_bullish_aligned[i] < 0.5 and 
                  volume_regime[i]):
                position = -1
                signals[i] = -0.25
    
    return signals