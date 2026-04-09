#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R + 1d Elder Ray (Bull/Bear Power) with volume confirmation
# - Williams %R(14) on 6h for overbought/oversold signals: < -80 = oversold, > -20 = overbought
# - Elder Ray on 1d: Bull Power = high - EMA(13), Bear Power = low - EMA(13)
# - Long when: Williams %R < -80 (oversold) AND Bull Power > 0 (bullish 1d trend) AND volume spike
# - Short when: Williams %R > -20 (overbought) AND Bear Power < 0 (bearish 1d trend) AND volume spike
# - Volume confirmation: 6h volume > 1.5x 20-period average
# - ATR(14) trailing stop at 2.0x ATR from extreme for risk control
# - Position size: 0.25 (25% of capital) - discrete level to minimize fee churn
# - Target: ~12-37 trades/year (50-150 total over 4 years) per 6h strategy guidelines
# - Novelty: Combines momentum oscillator (Williams %R) with trend strength (Elder Ray) on different timeframes
# - Works in both bull/bear: Elder Ray filters for 1d trend direction, Williams %R provides precise 6h entry timing

name = "6h_1d_williams_elderray_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 1d indicators
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA(13) for Elder Ray
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate 1d Elder Ray components
    bull_power_1d = high_1d - ema_13_1d  # > 0 indicates bullish strength
    bear_power_1d = low_1d - ema_13_1d   # < 0 indicates bearish strength
    
    # Align Elder Ray to 6h timeframe (completed 1d bar only)
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    # 6h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 6h Williams %R(14): (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = np.where(
        (highest_high_14 - lowest_low_14) > 0,
        ((highest_high_14 - close) / (highest_high_14 - lowest_low_14)) * -100,
        -50  # neutral when range is zero
    )
    williams_r = np.where(np.isnan(williams_r), -50, williams_r)
    
    # 6h volume > 1.5x 20-period average (volume confirmation)
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * avg_volume_20)
    
    # 6h ATR(14) for trailing stop
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or 
            np.isnan(bull_power_aligned[i]) or
            np.isnan(bear_power_aligned[i]) or
            np.isnan(volume_spike[i]) or
            np.isnan(atr[i]) or
            atr[i] <= 0):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Update highest high since entry
            if high[i] > highest_since_entry:
                highest_since_entry = high[i]
            
            # Exit conditions: price retraces 2.0x ATR from high OR Williams %R > -50 (momentum fading)
            if low[i] <= highest_since_entry - (2.0 * atr[i]) or \
               williams_r[i] > -50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest low since entry
            if low[i] < lowest_since_entry:
                lowest_since_entry = low[i]
            
            # Exit conditions: price retraces 2.0x ATR from low OR Williams %R < -50 (momentum fading)
            if high[i] >= lowest_since_entry + (2.0 * atr[i]) or \
               williams_r[i] < -50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Williams %R extreme with Elder Ray confirmation and volume spike
            # Long: Williams %R < -80 (oversold) AND Bull Power > 0 (bullish 1d trend) AND volume spike
            if williams_r[i] < -80 and bull_power_aligned[i] > 0 and volume_spike[i]:
                position = 1
                highest_since_entry = high[i]
                lowest_since_entry = high[i]
                signals[i] = 0.25
            # Short: Williams %R > -20 (overbought) AND Bear Power < 0 (bearish 1d trend) AND volume spike
            elif williams_r[i] > -20 and bear_power_aligned[i] < 0 and volume_spike[i]:
                position = -1
                highest_since_entry = low[i]
                lowest_since_entry = low[i]
                signals[i] = -0.25
    
    return signals