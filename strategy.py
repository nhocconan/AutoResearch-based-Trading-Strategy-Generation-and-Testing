#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d ADX25 regime filter and volume confirmation
# Elder Ray measures bull/bear power relative to EMA13: Bull Power = High - EMA13, Bear Power = Low - EMA13
# In trending markets (ADX > 25): go long when Bull Power > 0 and rising, short when Bear Power < 0 and falling
# In ranging markets (ADX <= 25): fade extremes - short when Bull Power > 0.5*ATR, long when Bear Power < -0.5*ATR
# Volume confirmation (1.5x 20-period average) filters weak breakouts. Works in both bull/bear by adapting to regime.
# Target: 12-37 trades/year on 6h.

name = "6h_ElderRay_1dADX25_Regime_Volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ADX regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # Calculate ADX(14) on 1d for regime filter
    # True Range
    tr1 = pd.Series(df_1d['high']).diff().abs()
    tr2 = (pd.Series(df_1d['high']) - pd.Series(df_1d['close'].shift(1))).abs()
    tr3 = (pd.Series(df_1d['low']) - pd.Series(df_1d['close'].shift(1))).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).values
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Directional Movement
    dm_plus = pd.Series(df_1d['high']).diff()
    dm_minus = -pd.Series(df_1d['low']).diff()
    dm_plus = np.where((dm_plus > dm_minus) & (dm_plus > 0), dm_plus, 0)
    dm_minus = np.where((dm_minus > dm_plus) & (dm_minus > 0), dm_minus, 0)
    
    # Smoothed DM and TR
    dm_plus_smooth = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    tr_smooth = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / tr_smooth
    di_minus = 100 * dm_minus_smooth / tr_smooth
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx_1d = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align ADX to 6h timeframe (wait for completed 1d bar)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate EMA(13) on 6h for Elder Ray
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Calculate ATR(14) on 6h for threshold scaling
    tr_6h = np.maximum(np.maximum(high - low, np.abs(high - np.roll(close, 1))), np.abs(low - np.roll(close, 1)))
    tr_6h[0] = high[0] - low[0]  # first bar
    atr_6h = pd.Series(tr_6h).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation (1.5x 20-period average) on 6h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for EMA, ATR, and volume calculations)
    start_idx = 30  # max(13 for EMA, 14 for ATR, 20 for volume +1 for shift)
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(ema_13[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(atr_6h[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Regime: ADX > 25 = trending, ADX <= 25 = ranging
        is_trending = adx_1d_aligned[i] > 25
        
        if position == 0:  # Flat - look for new entries
            if is_trending:
                # Trending regime: follow Elder Ray momentum
                # Long: Bull Power > 0 and rising (current > previous)
                # Short: Bear Power < 0 and falling (current < previous)
                if bull_power[i] > 0 and bull_power[i] > bull_power[i-1] and volume_spike[i]:
                    signals[i] = 0.25
                    position = 1
                elif bear_power[i] < 0 and bear_power[i] < bear_power[i-1] and volume_spike[i]:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                # Ranging regime: fade Elder Ray extremes
                # Short when Bull Power is excessively high
                # Long when Bear Power is excessively low
                if bull_power[i] > 0.5 * atr_6h[i] and volume_spike[i]:
                    signals[i] = -0.25
                    position = -1
                elif bear_power[i] < -0.5 * atr_6h[i] and volume_spike[i]:
                    signals[i] = 0.25
                    position = 1
                else:
                    signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit conditions
            if is_trending:
                # In trend: exit when Bull Power turns negative or momentum fades
                if bull_power[i] <= 0 or (i > 0 and bull_power[i] < bull_power[i-1]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:
                # In range: exit when price returns to neutral (EMA13) or opposite extreme
                if close[i] >= ema_13[i] or bear_power[i] < -0.3 * atr_6h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit conditions
            if is_trending:
                # In trend: exit when Bear Power turns positive or momentum fades
                if bear_power[i] >= 0 or (i > 0 and bear_power[i] > bear_power[i-1]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:
                # In range: exit when price returns to neutral (EMA13) or opposite extreme
                if close[i] <= ema_13[i] or bull_power[i] > 0.3 * atr_6h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals