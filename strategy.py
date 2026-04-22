#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Bollinger Bands squeeze breakout with 1d EMA34 trend filter and volume spike confirmation.
# This strategy exploits volatility contraction (Bollinger Bands squeeze) followed by breakout.
# The Bollinger Band Width (BBW) percentile identifies squeeze conditions (low volatility).
# When BBW is at a 20-period low and price breaks above/below the Bollinger Bands with volume confirmation (>1.5x 20-period average),
# it signals a high-probability breakout. The 1d EMA34 filter ensures we only take trades in the direction of the higher timeframe trend.
# Designed for low trade frequency (~20-30/year) to minimize fee decay. Works in both bull and bear markets by following higher timeframe trend.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for EMA34 calculation (once before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 34-period EMA on 1d close for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA to 4h timeframe (waits for 1d bar to close)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Bollinger Bands (20, 2) on 4h close
    close = prices['close'].values
    bb_mid = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    bb_std = pd.Series(close).rolling(window=20, min_periods=20).std().values
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    
    # Calculate Bollinger Band Width (BBW) and its 20-period percentile rank for squeeze detection
    bb_width = (bb_upper - bb_lower) / bb_mid
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=20, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else np.nan, raw=False
    ).values
    
    # Calculate 20-period average volume for volume spike detection
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema_34_aligned[i]) or 
            np.isnan(bb_mid[i]) or 
            np.isnan(bb_upper[i]) or 
            np.isnan(bb_lower[i]) or 
            np.isnan(bb_width_percentile[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        bb_width_pctl = bb_width_percentile[i]
        ema_val = ema_34_aligned[i]
        bb_up = bb_upper[i]
        bb_low = bb_lower[i]
        
        # Squeeze condition: BBW at 20-period low (bottom 20% of range)
        squeeze = bb_width_pctl <= 0.2
        
        # Volume filter: current volume > 1.5 * 20-period average
        vol_spike = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long conditions: squeeze breakout above upper BB + uptrend + volume spike
            if squeeze and price > bb_up and price > ema_val and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short conditions: squeeze breakout below lower BB + downtrend + volume spike
            elif squeeze and price < bb_low and price < ema_val and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: volatility expansion (end of squeeze) or opposite BB break
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when squeeze ends (volatility expands) or price breaks below lower BB
                if not squeeze or price < bb_low:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when squeeze ends (volatility expands) or price breaks above upper BB
                if not squeeze or price > bb_up:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_BollingerSqueeze_1dEMA34_Volume"
timeframe = "4h"
leverage = 1.0