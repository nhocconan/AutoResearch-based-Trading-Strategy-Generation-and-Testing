#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 12h/1d regime filter
# - Bull Power = High - EMA13(close), Bear Power = EMA13(close) - Low
# - Long when Bull Power > 0 AND Bear Power < 0 AND 12h close > 12h EMA26 (bullish regime)
# - Short when Bear Power > 0 AND Bull Power < 0 AND 12h close < 12h EMA26 (bearish regime)
# - Volume confirmation: 6h volume > 1.3x 20-period volume SMA
# - Exit: opposing Elder Ray signal or loss of volume confirmation
# - Position sizing: 0.25 discrete level to minimize fee drag
# - Target: 12-37 trades/year on 6h timeframe to stay within fee drag limits
# - Works in both bull/bear via regime filter (12h EMA26) and mean-reversion within trend via Elder Ray

name = "6h_12h_elderray_regime_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Calculate Elder Ray components: EMA13 of close
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13  # High - EMA13
    bear_power = ema13 - low   # EMA13 - Low
    
    # Calculate 12h EMA26 for regime filter
    close_12h = df_12h['close'].values
    ema26_12h = pd.Series(close_12h).ewm(span=26, adjust=False, min_periods=26).mean().values
    ema26_12h_aligned = align_htf_to_ltf(prices, df_12h, ema26_12h)
    
    # Calculate 12h close for regime comparison
    close_12h_aligned = align_htf_to_ltf(prices, df_12h, close_12h)
    
    # Calculate 6h volume SMA for confirmation
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(40, n):  # Start after warmup for indicators
        # Skip if any required data is invalid
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema26_12h_aligned[i]) or np.isnan(close_12h_aligned[i]) or
            np.isnan(volume_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: 6h volume > 1.3x 20-period volume SMA
        vol_confirm = volume[i] > 1.3 * volume_sma_20[i]
        
        # Regime filter: 12h close vs 12h EMA26
        regime_bullish = close_12h_aligned[i] > ema26_12h_aligned[i]
        regime_bearish = close_12h_aligned[i] < ema26_12h_aligned[i]
        
        # Elder Ray signals
        bull_signal = bull_power[i] > 0 and bear_power[i] < 0  # Bullish momentum
        bear_signal = bear_power[i] > 0 and bull_power[i] < 0  # Bearish momentum
        
        # Exit conditions: opposing Elder Ray signal or loss of volume confirmation
        exit_long = bear_signal or not vol_confirm
        exit_short = bull_signal or not vol_confirm
        
        if position == 0:  # Flat - look for entry
            if bull_signal and regime_bullish and vol_confirm:
                position = 1
                signals[i] = 0.25
            elif bear_signal and regime_bearish and vol_confirm:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        elif position == 1:  # Long position - look for exit
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        else:  # position == -1 (Short position) - look for exit
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
    
    return signals