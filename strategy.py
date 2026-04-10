#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d trend filter and volume confirmation
# - Bull Power = High - EMA13(close); Bear Power = EMA13(close) - Low
# - Long when Bull Power > 0 AND Bear Power < 0 AND price > 1d EMA50 (bullish trend) AND volume > 1.2x 20-period volume SMA
# - Short when Bear Power > 0 AND Bull Power < 0 AND price < 1d EMA50 (bearish trend) AND volume > 1.2x 20-period volume SMA
# - Exit when Elder Power signals reverse or volume drops below average
# - Position sizing: 0.25 discrete level to minimize fee drag
# - Target: 12-37 trades/year on 6h timeframe to stay within fee drag limits
# - Works in bull/bear via trend filter + momentum confirmation

name = "6h_1d_elderray_regime_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Calculate EMA13 for Elder Ray (primary timeframe)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13  # High - EMA13
    bear_power = ema13 - low   # EMA13 - Low
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 6h volume SMA for regime filter
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(40, n):  # Start after warmup for indicators
        # Skip if any required data is invalid
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: 6h volume > 1.2x 20-period volume SMA
        vol_confirm = volume[i] > 1.2 * volume_sma_20[i]
        
        # Trend filter: price vs 1d EMA50
        price_above_ema50 = close[i] > ema_50_1d_aligned[i]
        price_below_ema50 = close[i] < ema_50_1d_aligned[i]
        
        # Elder Ray signals
        bull_signal = bull_power[i] > 0 and bear_power[i] < 0  # Strong bullish momentum
        bear_signal = bear_power[i] > 0 and bull_power[i] < 0  # Strong bearish momentum
        
        # Exit conditions: reverse Elder Ray signal or loss of volume confirmation
        exit_long = (not bull_signal) or (not vol_confirm)
        exit_short = (not bear_signal) or (not vol_confirm)
        
        if position == 0:  # Flat - look for entry
            if bull_signal and price_above_ema50 and vol_confirm:
                position = 1
                signals[i] = 0.25
            elif bear_signal and price_below_ema50 and vol_confirm:
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