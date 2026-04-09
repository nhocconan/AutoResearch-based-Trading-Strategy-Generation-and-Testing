#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R + Elder Ray (Bull/Bear Power) + 1d volume regime filter
# - Williams %R(14) on 6h: oversold < -80 for long, overbought > -20 for short
# - Elder Ray: Bull Power = high - EMA(13), Bear Power = low - EMA(13) on 6h
#   Long when Bull Power > 0 and rising, Short when Bear Power < 0 and falling
# - 1d volume regime: only trade when 1d volume > 20-period average (avoid low-volume whipsaws)
# - Position size: 0.25 (discrete level to minimize fee churn)
# - ATR(14) trailing stop at 2.0x ATR from extreme for risk control
# - Works in bull/bear: Williams %R captures reversals, Elder Ray confirms trend strength,
#   volume filter ensures participation only during active market phases
# - Target: ~12-37 trades/year (50-150 total over 4 years) per 6h strategy guidelines

name = "6h_1d_williams_elderray_volume_v2"
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
    volume_1d = df_1d['volume'].values
    
    # 1d volume > 20-period average (volume regime filter)
    avg_volume_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_regime = volume_1d > avg_volume_20
    
    # Align volume regime to 6h timeframe (completed 1d bar only)
    volume_regime_aligned = align_htf_to_ltf(prices, df_1d, volume_regime)
    
    # 6h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 6h Williams %R(14): %R = (highest_high - close) / (highest_high - lowest_low) * -100
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = np.where((highest_high_14 - lowest_low_14) > 0,
                          ((highest_high_14 - close) / (highest_high_14 - lowest_low_14)) * -100, -50)
    williams_r = np.where(np.isnan(williams_r), -50, williams_r)
    
    # 6h EMA(13) for Elder Ray
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # 6h Elder Ray: Bull Power = high - EMA, Bear Power = low - EMA
    bull_power = high - ema_13
    bear_power = low - ema_13
    
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
            np.isnan(bull_power[i]) or
            np.isnan(bear_power[i]) or
            np.isnan(volume_regime_aligned[i]) or
            np.isnan(atr[i]) or
            atr[i] <= 0):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Update highest high since entry
            if high[i] > highest_since_entry:
                highest_since_entry = high[i]
            
            # Exit conditions: price retraces 2.0x ATR from high OR Williams %R > -20 (overbought)
            if low[i] <= highest_since_entry - (2.0 * atr[i]) or \
               williams_r[i] > -20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest low since entry
            if low[i] < lowest_since_entry:
                lowest_since_entry = low[i]
            
            # Exit conditions: price retraces 2.0x ATR from low OR Williams %R < -80 (oversold)
            if high[i] >= lowest_since_entry + (2.0 * atr[i]) or \
               williams_r[i] < -80:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for entry: Williams %R extreme + Elder Ray confirmation + volume regime
            # Long: Williams %R < -80 (oversold) AND Bull Power > 0 AND Bull Power rising AND volume regime
            if (williams_r[i] < -80 and 
                bull_power[i] > 0 and 
                i > 100 and bull_power[i] > bull_power[i-1] and
                volume_regime_aligned[i]):
                position = 1
                highest_since_entry = high[i]
                lowest_since_entry = high[i]
                signals[i] = 0.25
            # Short: Williams %R > -20 (overbought) AND Bear Power < 0 AND Bear Power falling AND volume regime
            elif (williams_r[i] > -20 and 
                  bear_power[i] < 0 and 
                  i > 100 and bear_power[i] < bear_power[i-1] and
                  volume_regime_aligned[i]):
                position = -1
                highest_since_entry = low[i]
                lowest_since_entry = low[i]
                signals[i] = -0.25
    
    return signals