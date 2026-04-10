#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) + 1d ADX trend filter + volume confirmation
# - Elder Ray: Bull Power = High - EMA(13), Bear Power = Low - EMA(13)
# - Long when Bull Power > 0 AND Bear Power rising (less negative) AND 1d ADX > 25 (trending) AND volume > 1.5x 20-period average
# - Short when Bear Power < 0 AND Bull Power falling (less positive) AND 1d ADX > 25 (trending) AND volume > 1.5x 20-period average
# - Exit when Elder Power signals reverse or ADX < 20 (range market)
# - Uses discrete position sizing 0.25 to limit fee churn
# - Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years)
# - Works in trending markets by measuring bull/bear strength relative to EMA
# - ADX filter ensures we only trade in strong trends where Elder Ray signals are reliable

name = "6h_1d_elder_ray_adx_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 6h Elder Ray components
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # EMA(13) for Elder Ray
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA(13)
    bull_power = high - ema_13
    
    # Bear Power = Low - EMA(13)
    bear_power = low - ema_13
    
    # Pre-compute 6h volume confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Pre-compute 1d ADX (trend strength filter)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - np.roll(close_1d, 1)[1:])
    tr3 = np.abs(low_1d[1:] - np.roll(close_1d, 1)[1:])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # first element is NaN
    
    # ATR(14) for ADX calculation
    atr_14 = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
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
    atr_smooth = pd.Series(atr_14).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / np.where(atr_smooth != 0, atr_smooth, 1e-10)
    minus_di = 100 * minus_dm_smooth / np.where(atr_smooth != 0, atr_smooth, 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / np.where((plus_di + minus_di) != 0, (plus_di + minus_di), 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align HTF indicators to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(adx_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: Bull Power positive AND Bear Power rising (less negative) AND ADX > 25 AND volume spike
            if (bull_power[i] > 0 and 
                bear_power[i] > bear_power[i-1] and  # Bear Power rising (less negative)
                adx_aligned[i] > 25 and 
                volume_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short conditions: Bear Power negative AND Bull Power falling (less positive) AND ADX > 25 AND volume spike
            elif (bear_power[i] < 0 and 
                  bull_power[i] < bull_power[i-1] and  # Bull Power falling (less positive)
                  adx_aligned[i] > 25 and 
                  volume_spike[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit when Elder Power signals reverse OR ADX < 20 (range market)
            exit_long = (position == 1 and 
                        (bull_power[i] <= 0 or  # Bull Power no longer positive
                         bear_power[i] <= bear_power[i-1] or  # Bear Power stopped rising
                         adx_aligned[i] < 20))  # ADX dropped below 20 (range)
            exit_short = (position == -1 and 
                         (bear_power[i] >= 0 or  # Bear Power no longer negative
                          bull_power[i] >= bull_power[i-1] or  # Bull Power stopped falling
                          adx_aligned[i] < 20))  # ADX dropped below 20 (range)
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals