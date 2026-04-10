#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + ADX regime + volume confirmation
# - Bull Power = EMA(13) of (High - EMA(20)) measures bull strength
# - Bear Power = EMA(13) of (Low - EMA(20)) measures bear strength
# - ADX(14) > 25 indicates trending regime for Elder Ray signals
# - Long when Bull Power > 0 AND Bear Power < 0 AND ADX > 25 AND volume > 1.5x 20-period average
# - Short when Bull Power < 0 AND Bear Power > 0 AND ADX > 25 AND volume > 1.5x 20-period average
# - Exit when Elder Ray signal weakens (Bull Power < 0 for longs, Bear Power > 0 for shorts)
# - Uses discrete position sizing 0.25 to limit fee churn
# - Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years)
# - Elder Ray identifies power imbalance between bulls and bears
# - ADX filter ensures we trade only in trending markets where Elder Ray works best
# - Volume confirmation reduces false signals

name = "6h_12h_elder_ray_adx_volume_v1"
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
    
    # Pre-compute OHLC and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Calculate EMA(20) for Elder Ray
    close_s = pd.Series(close)
    ema_20 = close_s.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate Elder Ray components
    bull_power_raw = high - ema_20  # Bull Power = High - EMA
    bear_power_raw = low - ema_20   # Bear Power = Low - EMA
    
    # Smooth with EMA(13) as per Elder Ray specification
    bull_power = pd.Series(bull_power_raw).ewm(span=13, adjust=False, min_periods=13).mean().values
    bear_power = pd.Series(bear_power_raw).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate ADX(14) for regime filter
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(14) using Wilder's smoothing
    atr = np.zeros_like(tr)
    atr[13] = np.mean(tr[1:14])  # First ATR value
    for i in range(14, len(tr)):
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # Directional Movement
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    up_move[0] = 0
    down_move[0] = 0
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed DM and ATR
    plus_dm_smooth = np.zeros_like(plus_dm)
    minus_dm_smooth = np.zeros_like(minus_dm)
    atr_smooth = np.zeros_like(atr)
    
    # First values (simple average)
    plus_dm_smooth[13] = np.mean(plus_dm[1:14])
    minus_dm_smooth[13] = np.mean(minus_dm[1:14])
    atr_smooth[13] = np.mean(atr[1:14])
    
    # Wilder's smoothing
    for i in range(14, len(tr)):
        plus_dm_smooth[i] = (plus_dm_smooth[i-1] * 13 + plus_dm[i]) / 14
        minus_dm_smooth[i] = (minus_dm_smooth[i-1] * 13 + minus_dm[i]) / 14
        atr_smooth[i] = (atr_smooth[i-1] * 13 + tr[i]) / 14
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / atr_smooth
    minus_di = 100 * minus_dm_smooth / atr_smooth
    
    # DX and ADX
    dx = np.zeros_like(atr_smooth)
    dx[13:] = 100 * np.abs(plus_di[13:] - minus_di[13:]) / (plus_di[13:] + minus_di[13:])
    
    adx = np.zeros_like(dx)
    adx[26] = np.mean(dx[14:28])  # First ADX value
    for i in range(27, len(dx)):
        adx[i] = (adx[i-1] * 13 + dx[i]) / 14
    
    # ADX regime: trending when ADX > 25
    trending_regime = adx > 25
    
    # Align HTF indicators to 6h timeframe (not used in this version but keeping structure)
    # df_12h is loaded but not used for HTF indicators in this strategy
    # Keeping the structure for potential future MTF enhancements
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(adx[i]) or 
            np.isnan(vol_ma[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: Bull Power > 0 AND Bear Power < 0 AND trending regime AND volume spike
            if (bull_power[i] > 0 and 
                bear_power[i] < 0 and 
                trending_regime[i] and 
                volume_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short conditions: Bull Power < 0 AND Bear Power > 0 AND trending regime AND volume spike
            elif (bull_power[i] < 0 and 
                  bear_power[i] > 0 and 
                  trending_regime[i] and 
                  volume_spike[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit when Elder Ray signal weakens
            # Exit when Elder Ray signal weakens (Bull Power < 0 for longs, Bear Power > 0 for shorts)
            exit_long = (position == 1 and bull_power[i] < 0)
            exit_short = (position == -1 and bear_power[i] > 0)
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals