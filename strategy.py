#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA(50) trend filter and volume confirmation.
# Long when close > Donchian(20) high AND price > 1w EMA(50) AND volume > 1.5x 20-period average.
# Short when close < Donchian(20) low AND price < 1w EMA(50) AND volume > 1.5x 20-period average.
# Uses discrete position size 0.25. Donchian captures breakouts, 1w EMA ensures we trade with higher timeframe trend (avoiding counter-trend whipsaws),
# volume spike confirms institutional participation. Designed to work in both bull (buy breakouts) and bear (sell breakdowns) markets.
# Target: 30-100 trades over 4 years (7-25/year) to minimize fee drag while capturing major moves.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d Indicators: Donchian(20) ===
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 1d Indicators: Volume Spike (volume > 1.5x 20-period average) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Get 1w data once before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:  # Need enough for EMA calculation
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # === 1w Indicators: EMA(50) for trend filter ===
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 60 periods needed for 1w EMA, 20 for Donchian/volume)
    warmup = 60
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        upper = donchian_high[i]
        lower = donchian_low[i]
        ema_50 = ema_50_1w_aligned[i]
        vol_spike = volume_spike[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price breaks below Donchian low or volume spike ends
            if price < lower or not vol_spike:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price breaks above Donchian high or volume spike ends
            if price > upper or not vol_spike:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: price > Donchian high AND price > 1w EMA(50) AND volume spike
            if price > upper and price > ema_50 and vol_spike:
                signals[i] = 0.25
                position = 1
            
            # SHORT: price < Donchian low AND price < 1w EMA(50) AND volume spike
            elif price < lower and price < ema_50 and vol_spike:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "1d_Donchian20_1wEMA50_VolumeSpike_V1"
timeframe = "1d"
leverage = 1.0