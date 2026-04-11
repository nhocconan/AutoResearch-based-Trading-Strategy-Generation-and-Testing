#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 12h volume confirmation and 1d ADX trend filter
# - Enter long when price breaks above Donchian(20) upper band AND 12h volume > 1.5x 20-period volume SMA AND 1d ADX > 25
# - Enter short when price breaks below Donchian(20) lower band AND 12h volume > 1.5x 20-period volume SMA AND 1d ADX > 25
# - Exit: ATR-based trailing stop (3x ATR) or opposite Donchian band touch
# - Donchian provides clear structure breakouts
# - Volume confirmation ensures institutional participation
# - ADX filter avoids whipsaws in ranging markets
# - Target: 20-50 trades/year to minimize fee drag while capturing strong trends

name = "4h_12h_1d_donchian_volume_adx_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    atr = np.zeros(n)  # ATR for trailing stop
    
    # Load 12h data ONCE before loop for volume confirmation (MTF rule compliance)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return signals
    
    # Load 1d data ONCE before loop for ADX trend filter (MTF rule compliance)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return signals
    
    # Pre-compute Donchian channels for 4h data (20-period)
    # Upper band = highest high over 20 periods
    # Lower band = lowest low over 20 periods
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    donchian_upper = high_s.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_s.rolling(window=20, min_periods=20).min().values
    
    # Pre-compute ATR for 4h data (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Pre-compute volume SMA for 12h data (20-period)
    volume_12h = df_12h['volume'].values
    volume_sma_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_sma_20_12h)
    
    # Pre-compute ADX for 1d data (14-period)
    # ADX calculation: +DI, -DI, DX, then ADX as smoothed DX
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1_1d = high_1d - low_1d
    tr2_1d = np.abs(high_1d - np.roll(close_1d, 1))
    tr3_1d = np.abs(low_1d - np.roll(close_1d, 1))
    tr2_1d[0] = 0
    tr3_1d[0] = 0
    tr_1d = np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    tr_14 = pd.Series(tr_1d).rolling(window=14, min_periods=14).sum().values
    dm_plus_14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).sum().values
    dm_minus_14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).sum().values
    
    # DI values
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align indicators to 4h timeframe
    volume_sma_20_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_sma_20_12h)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
    
    # Pre-compute 12h volume for confirmation
    volume_12h_current = df_12h['volume'].values
    volume_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_12h_current)
    
    for i in range(20, n):  # Start after 20-bar warmup for Donchian
        # Skip if any required data is invalid
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or np.isnan(atr[i]) or
            np.isnan(volume_sma_20_12h_aligned[i]) or np.isnan(adx_aligned[i]) or
            np.isnan(volume_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: 12h volume > 1.5x 20-period volume SMA
        vol_confirm = volume_12h_aligned[i] > 1.5 * volume_sma_20_12h_aligned[i]
        
        # Trend filter: 1d ADX > 25 (strong trend)
        trend_filter = adx_aligned[i] > 25
        
        # Donchian breakout signals
        breakout_up = close[i] > donchian_upper[i-1]  # Break above upper band (using previous close to avoid look-ahead)
        breakout_down = close[i] < donchian_lower[i-1]  # Break below lower band
        
        # Exit conditions
        # Trailing stop: highest high since entry minus 3*ATR for longs, lowest low plus 3*ATR for shorts
        # Opposite Donchian band touch
        exit_long = False
        exit_short = False
        
        if position == 1:  # Long position
            # Trailing stop logic (simplified: exit if price drops 3*ATR from entry)
            # We'll use opposite Donchian touch as primary exit for simplicity
            exit_long = close[i] < donchian_lower[i]  # Exit long when price touches lower band
        elif position == -1:  # Short position
            exit_short = close[i] > donchian_upper[i]  # Exit short when price touches upper band
        
        # Trading logic
        if vol_confirm and trend_filter:
            # Long: Donchian breakout up
            if breakout_up:
                if position != 1:  # Only signal on new long entry
                    position = 1
                    signals[i] = 0.25
                else:
                    signals[i] = 0.25
            # Short: Donchian breakout down
            elif breakout_down:
                if position != -1:  # Only signal on new short entry
                    position = -1
                    signals[i] = -0.25
                else:
                    signals[i] = -0.25
            else:
                # Check for exits
                if position == 1 and exit_long:
                    position = 0
                    signals[i] = 0.0
                elif position == -1 and exit_short:
                    position = 0
                    signals[i] = 0.0
                else:
                    # Maintain current position
                    signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
        else:
            # No volume confirmation or no trend: exit any position
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
    
    return signals