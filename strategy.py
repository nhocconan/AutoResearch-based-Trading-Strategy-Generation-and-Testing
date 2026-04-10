#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Elder Ray Bull/Bear Power + 1d ADX regime filter + volume confirmation
# - Bull Power = High - EMA(13); Bear Power = EMA(13) - Low
# - Long when Bull Power > 0 AND Bear Power < 0 (both bullish) AND volume > 1.5x 20-period average AND 1d ADX < 25 (ranging)
# - Short when Bear Power > 0 AND Bull Power < 0 (both bearish) AND volume > 1.5x 20-period average AND 1d ADX < 25 (ranging)
# - Exit when either power crosses zero OR volume drops
# - Uses discrete position sizing 0.25 to limit fee churn
# - Target: 19-50 trades/year on 4h timeframe (75-200 total over 4 years)
# - Elder Ray measures bull/bear strength relative to EMA
# - Volume confirmation ensures conviction
# - ADX < 25 filter ensures we trade in ranging/low trend conditions where Elder Ray works best

name = "4h_1d_elder_ray_volume_adx_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 4h Elder Ray components
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # EMA(13) for Elder Ray
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA(13)
    bull_power = high - ema13
    
    # Bear Power = EMA(13) - Low
    bear_power = ema13 - low
    
    # Pre-compute 4h volume confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Pre-compute 1d ADX (14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - np.roll(close_1d, 1)[1:])
    tr3 = np.abs(low_1d[1:] - np.roll(close_1d, 1)[1:])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # first element is NaN
    
    # ATR(14)
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Directional Movement
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    plus_dm = np.concatenate([[0.0], plus_dm])
    minus_dm = np.concatenate([[0.0], minus_dm])
    
    # Smoothed DM and ATR
    plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    atr_smooth = pd.Series(atr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # DI+ and DI-
    plus_di = 100 * plus_dm_smooth / atr_smooth
    minus_di = 100 * minus_dm_smooth / atr_smooth
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    dx = np.where((plus_di + minus_di) == 0, 0, dx)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # ADX regime: < 25 = ranging/low trend (good for Elder Ray mean reversion)
    adx_regime = adx < 25
    
    # Align HTF indicators to 4h timeframe
    adx_regime_aligned = align_htf_to_ltf(prices, df_1d, adx_regime)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(adx_regime_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: Bull Power > 0 AND Bear Power < 0 (both bullish) AND volume spike AND ADX < 25
            if (bull_power[i] > 0 and 
                bear_power[i] < 0 and 
                volume_spike[i] and 
                adx_regime_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short conditions: Bear Power > 0 AND Bull Power < 0 (both bearish) AND volume spike AND ADX < 25
            elif (bear_power[i] > 0 and 
                  bull_power[i] < 0 and 
                  volume_spike[i] and 
                  adx_regime_aligned[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit when either power crosses zero OR volume drops below average
            exit_long = (position == 1 and 
                        (bull_power[i] <= 0 or bear_power[i] >= 0 or not volume_spike[i]))
            exit_short = (position == -1 and 
                         (bear_power[i] <= 0 or bull_power[i] >= 0 or not volume_spike[i]))
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals