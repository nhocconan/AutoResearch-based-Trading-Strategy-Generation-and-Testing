#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h volume confirmation and ATR regime filter
# - Primary: 4h timeframe for balance of trade frequency and fee drag
# - HTF: 12h for volume confirmation and ATR volatility regime
# - Long: Price breaks above Donchian(20) high + 12h ATR > 50th percentile + volume > 1.5x 20-period MA
# - Short: Price breaks below Donchian(20) low + 12h ATR > 50th percentile + volume > 1.5x 20-period MA
# - Exit: ATR-based trailing stop (3x ATR) or Donchian(10) opposite breakout
# - Position sizing: 0.30 (discrete level)
# - Target: 80-180 total trades over 4 years (20-45/year) - within 4h sweet spot
# - Works in bull/bear: Donchian breakouts capture trends, ATR regime avoids chop, volume confirms strength

name = "4h_12h_donchian_volume_atr_v5"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need enough data for calculations
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
    
    # Calculate 4h Donchian channels (20-period)
    donchian_high_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_high_10 = pd.Series(high_4h).rolling(window=10, min_periods=10).max().values  # For exit
    donchian_low_10 = pd.Series(low_4h).rolling(window=10, min_periods=10).min().values   # For exit
    
    # Calculate 12h ATR(14) for volatility regime filter
    tr1 = pd.Series(high_12h).shift(1) - pd.Series(low_12h).shift(1)
    tr2 = abs(pd.Series(high_12h) - pd.Series(close_12h).shift(1))
    tr3 = abs(pd.Series(low_12h) - pd.Series(close_12h).shift(1))
    tr_12h = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_12h = tr_12h.rolling(window=14, min_periods=14).mean().values
    
    # Calculate 12h ATR percentile rank (using 30-bar lookback)
    atr_percentile = pd.Series(atr_12h).rolling(window=30, min_periods=10).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    atr_percentile_aligned = align_htf_to_ltf(prices, df_12h, atr_percentile)
    
    # Calculate 12h volume moving average (20-period) for volume confirmation
    volume_ma_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    volume_ma_20_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_ma_20_12h)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0  # Track entry price for ATR stop calculation
    
    for i in range(30, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(donchian_high_20[i]) or np.isnan(donchian_low_20[i]) or 
            np.isnan(donchian_high_10[i]) or np.isnan(donchian_low_10[i]) or
            np.isnan(atr_percentile_aligned[i]) or 
            np.isnan(volume_ma_20_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime conditions
        # 12h volatility regime: ATR > 50th percentile (avoid low-vol chop)
        vol_regime = atr_percentile_aligned[i] > 50
        
        # Volume confirmation: current 12h volume > 1.5x 20-period MA
        volume_spike = volume_12h[i] > 1.5 * volume_ma_20_12h_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Price breaks above Donchian(20) high + vol regime + volume spike
            if (close_4h[i] > donchian_high_20[i] and vol_regime and volume_spike):
                position = 1
                entry_price = close_4h[i]  # Approximate entry price (next bar open)
                signals[i] = 0.30
            # Short entry: Price breaks below Donchian(20) low + vol regime + volume spike
            elif (close_4h[i] < donchian_low_20[i] and vol_regime and volume_spike):
                position = -1
                entry_price = close_4h[i]  # Approximate entry price (next bar open)
                signals[i] = -0.30
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Calculate ATR-based stop level
            atr_value = atr_12h[i]  # Current 12h ATR
            atr_stop_distance = 3.0 * atr_value  # 3x ATR stop
            
            if position == 1:  # Long position
                # Exit conditions:
                # 1. ATR trailing stop: price < highest high since entry - 3*ATR
                # 2. Donchian(10) opposite breakout: price < Donchian low(10)
                
                # Track highest high since entry (simplified: use rolling max of recent highs)
                # For simplicity, we'll use a trailing stop based on ATR from current levels
                long_stop = high_4h[i] - atr_stop_distance  # Simplified trailing stop
                
                exit_condition = (
                    close_4h[i] < long_stop or  # ATR trailing stop hit
                    close_4h[i] < donchian_low_10[i]  # Donchian(10) breakout
                )
                if exit_condition:
                    position = 0
                    entry_price = 0.0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.30
            else:  # position == -1 (Short position)
                # Exit conditions:
                # 1. ATR trailing stop: price > lowest low since entry + 3*ATR
                # 2. Donchian(10) opposite breakout: price > Donchian high(10)
                
                # Track lowest low since entry (simplified: use rolling min of recent lows)
                short_stop = low_4h[i] + atr_stop_distance  # Simplified trailing stop
                
                exit_condition = (
                    close_4h[i] > short_stop or  # ATR trailing stop hit
                    close_4h[i] > donchian_high_10[i]  # Donchian(10) breakout
                )
                if exit_condition:
                    position = 0
                    entry_price = 0.0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.30
    
    return signals