#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + volume confirmation + 12h ADX regime filter
# - Donchian breakout captures momentum in trending markets (works in both bull/bear via direction filter)
# - Volume confirmation ensures institutional participation (reduces false breakouts)
# - 12h ADX > 20 ensures we only trade in non-choppy markets (avoids whipsaws)
# - Position sizing: 0.25 (discrete level to minimize fee churn)
# - Target: 100-200 total trades over 4 years (25-50/year)
# - Donchian channels provide objective breakout levels with clear stoploss logic
# - Volume filter adds confirmation without excessive trading
# - ADX regime filter prevents trading in choppy conditions where breakouts fail

name = "4h_12h_donchian_volume_adx_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:  # Need enough data for calculations
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Pre-compute 4h OHLCV
    open_4h = prices['open'].values
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    volume_4h = prices['volume'].values
    
    # Pre-compute 12h data
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate 4h Donchian(20) channels
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Calculate 12h ADX(14) for regime filter
    # True Range calculation
    tr1 = pd.Series(high_12h).shift(1) - pd.Series(low_12h).shift(1)
    tr2 = abs(pd.Series(high_12h) - pd.Series(close_12h).shift(1))
    tr3 = abs(pd.Series(low_12h) - pd.Series(close_12h).shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_12h = tr.rolling(window=14, min_periods=14).mean().values
    
    up_move = pd.Series(high_12h).diff()
    down_move = -pd.Series(low_12h).diff()
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    plus_di_12h = 100 * (pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr_12h)
    minus_di_12h = 100 * (pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr_12h)
    dx_12h = 100 * abs(plus_di_12h - minus_di_12h) / (plus_di_12h + minus_di_12h)
    adx_12h = pd.Series(dx_12h).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Handle division by zero in ADX calculation
    adx_12h = np.where((plus_di_12h + minus_di_12h) == 0, 0, adx_12h)
    
    # Align 12h indicators to 4h
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # Calculate 4h volume moving average (20-period)
    volume_ma_20_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(adx_aligned[i]) or 
            np.isnan(volume_ma_20_4h[i])):
            signals[i] = 0.0
            continue
        
        # Get current values
        high_current = high_4h[i]
        low_current = low_4h[i]
        close_current = close_4h[i]
        volume_current = volume_4h[i]
        adx_current = adx_aligned[i]
        volume_ma_current = volume_ma_20_4h[i]
        
        # Regime condition: non-choppy market (ADX > 20)
        non_choppy = adx_current > 20
        
        # Volume confirmation: current volume > 1.5x 20-period MA
        volume_confirmed = volume_current > 1.5 * volume_ma_current
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above Donchian high + non-choppy + volume confirmation
            if (high_current > donchian_high[i] and non_choppy and volume_confirmed):
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below Donchian low + non-choppy + volume confirmation
            elif (low_current < donchian_low[i] and non_choppy and volume_confirmed):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit when price returns to middle of Donchian channel
            donchian_mid = (donchian_high[i] + donchian_low[i]) / 2
            if position == 1:
                if close_current <= donchian_mid:  # Exit long when price returns to midpoint
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close_current >= donchian_mid:  # Exit short when price returns to midpoint
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals