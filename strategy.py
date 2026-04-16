#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Bollinger Band breakout with 4h trend filter and volume confirmation.
# Long when price breaks above upper BB(20,2) AND 4h EMA50 uptrend (price > EMA50) AND volume > 1.3x 20-period average.
# Short when price breaks below lower BB(20,2) AND 4h EMA50 downtrend (price < EMA50) AND volume > 1.3x 20-period average.
# Uses discrete position size 0.20. BB breakout captures momentum, 4h EMA50 ensures alignment with higher timeframe trend,
# volume spike confirms participation. Session filter (08-20 UTC) reduces noise. Designed to work in both bull and bear markets.
# Target: 80-150 trades over 4 years (20-37/year) to balance opportunity and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # === 1h Indicators: Bollinger Bands (20,2) ===
    close_s = pd.Series(close)
    basis = close_s.rolling(window=20, min_periods=20).mean().values
    dev = close_s.rolling(window=20, min_periods=20).std().values
    upper_bb = basis + 2 * dev
    lower_bb = basis - 2 * dev
    
    # === 1h Indicators: Volume Spike (volume > 1.3x 20-period average) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.3 * vol_ma)
    
    # Get 4h data once before loop for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:  # Need enough for EMA50 calculation
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # === 4h Indicators: EMA50 for trend filter ===
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 4h EMA50 to 1h timeframe
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 50 periods needed for EMA, 20 for BB/volume MA)
    warmup = 60
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(basis[i]) or np.isnan(ema_50_4h_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Session filter: only trade during 08-20 UTC
        if not in_session[i]:
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        upper = upper_bb[i]
        lower = lower_bb[i]
        ema_4h = ema_50_4h_aligned[i]
        vol_spike = volume_spike[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price crosses below basis (mean reversion) or volume spike ends
            if price < basis[i] or not vol_spike:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price crosses above basis (mean reversion) or volume spike ends
            if price > basis[i] or not vol_spike:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: price breaks above upper BB AND 4h EMA50 uptrend AND volume spike
            if price > upper and price > ema_4h and vol_spike:
                signals[i] = 0.20
                position = 1
            
            # SHORT: price breaks below lower BB AND 4h EMA50 downtrend AND volume spike
            elif price < lower and price < ema_4h and vol_spike:
                signals[i] = -0.20
                position = -1
        
        else:
            signals[i] = position * 0.20
    
    return signals

name = "1h_BB_Breakout_4hEMA50_VolumeSpike_Session_V1"
timeframe = "1h"
leverage = 1.0