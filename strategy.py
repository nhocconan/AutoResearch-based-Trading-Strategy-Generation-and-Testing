#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily Williams %R mean reversion with weekly trend filter and volume confirmation.
# Long when Williams %R(14) < -80 (oversold) AND weekly close > weekly EMA34 (bullish trend) AND daily volume > 1.5x 20-day average.
# Short when Williams %R(14) > -20 (overbought) AND weekly close < weekly EMA34 (bearish trend) AND daily volume > 1.5x 20-day average.
# Uses discrete position size 0.25. Williams %R captures extreme reversals, weekly EMA34 ensures we trade with higher timeframe trend (avoiding counter-trend whipsaws),
# volume spike confirms institutional participation. Designed to work in both bull (buy dips) and bear (sell rallies) markets by aligning with weekly trend.
# Target: 40-80 trades over 4 years (10-20/year) to minimize fee drag while capturing meaningful reversals.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === Daily Indicators: Williams %R(14) ===
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min()
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    williams_r_values = williams_r.values
    
    # === Daily Indicators: Volume Spike (volume > 1.5x 20-period average) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_spike = volume > (1.5 * vol_ma.values)
    
    # Get weekly data once before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 40:  # Need enough for EMA calculation
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # === Weekly Indicators: EMA(34) for trend filter ===
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean()
    ema_34_1w_values = ema_34_1w.values
    
    # Align weekly EMA to daily timeframe
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w_values)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 34 periods needed for weekly EMA, 20 for volume MA, 14 for Williams %R)
    warmup = 40
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r_values[i]) or np.isnan(ema_34_1w_aligned[i]) or
            np.isnan(vol_ma.iloc[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        williams_val = williams_r_values[i]
        ema_trend = ema_34_1w_aligned[i]
        vol_spike = volume_spike.iloc[i]
        price = close[i]
        weekly_close = close_1w[df_1w.index.get_indexer([prices.index[i]], method='pad')[0]] if hasattr(df_1w.index, 'get_indexer') else close_1w[-1]  # simplified for alignment check
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if Williams %R returns to neutral (-50) or weekly trend turns bearish
            if williams_val >= -50 or price < ema_34_1w_aligned[i]:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if Williams %R returns to neutral (-50) or weekly trend turns bullish
            if williams_val <= -50 or price > ema_34_1w_aligned[i]:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Williams %R < -80 (oversold) AND weekly close > weekly EMA34 (bullish trend) AND volume spike
            if williams_val < -80 and price > ema_34_1w_aligned[i] and vol_spike:
                signals[i] = 0.25
                position = 1
            
            # SHORT: Williams %R > -20 (overbought) AND weekly close < weekly EMA34 (bearish trend) AND volume spike
            elif williams_val > -20 and price < ema_34_1w_aligned[i] and vol_spike:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "1d_WilliamsR14_1wEMA34_VolumeSpike_V1"
timeframe = "1d"
leverage = 1.0