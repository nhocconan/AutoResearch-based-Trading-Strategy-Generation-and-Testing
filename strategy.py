#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band breakout with 1w trend filter and volume confirmation.
# Long when price breaks above upper BB(20,2) AND 1w close > 1w EMA50 (uptrend) AND 6h volume > 2.0x 20-period average.
# Short when price breaks below lower BB(20,2) AND 1w close < 1w EMA50 (downtrend) AND 6h volume > 2.0x 20-period average.
# Uses discrete position size 0.25. Bollinger Bands identify volatility expansion, 1w EMA50 ensures alignment with higher timeframe trend,
# extreme volume spike confirms institutional participation. Designed to capture strong momentum moves in both bull and bear markets.
# Target: 80-160 trades over 4 years (20-40/year) to balance opportunity and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 6h Indicators: Bollinger Bands (20,2) ===
    close_s = pd.Series(close)
    bb_ma = close_s.rolling(window=20, min_periods=20).mean().values
    bb_std = close_s.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_ma + 2 * bb_std
    bb_lower = bb_ma - 2 * bb_std
    
    # === 6h Indicators: Volume Spike (volume > 2.0x 20-period average) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Get 1w data once before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:  # Need enough for EMA50 calculation
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # === 1w Indicators: EMA50 for trend filter ===
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1w EMA50 to 6h timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 50 periods needed for EMA, 20 for BB/volume MA)
    warmup = 60
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        bb_up = bb_upper[i]
        bb_low = bb_lower[i]
        ema_1w = ema_50_1w_aligned[i]
        vol_spike = volume_spike[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price returns to middle band or volume spike ends
            if price <= bb_ma[i] or not vol_spike:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price returns to middle band or volume spike ends
            if price >= bb_ma[i] or not vol_spike:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above upper BB AND 1w close > 1w EMA50 (uptrend) AND volume spike
            if price > bb_up and close_1w[-1] > ema_1w and vol_spike:
                signals[i] = 0.25
                position = 1
            
            # SHORT: Price breaks below lower BB AND 1w close < 1w EMA50 (downtrend) AND volume spike
            elif price < bb_low and close_1w[-1] < ema_1w and vol_spike:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "6h_BollingerBreakout_1wEMA50_VolumeSpike_V1"
timeframe = "6h"
leverage = 1.0