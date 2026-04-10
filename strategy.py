#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 1d ADX regime filter + volume confirmation
# - Elder Ray: Bull Power = High - EMA13, Bear Power = EMA13 - Low (measures bull/bear strength)
# - ADX filter: Only trade when ADX > 25 on 1d (trending market) to avoid chop
# - Volume: 6h volume > 1.3x 20-period average for momentum confirmation
# - Entry: Long when Bull Power > 0 AND Bear Power < 0 AND ADX > 25 AND volume spike
#          Short when Bear Power > 0 AND Bull Power < 0 AND ADX > 25 AND volume spike
# - Exit: Opposite Elder Ray signal OR stoploss at 2.0x ATR(14)
# - Position size: 0.25 discrete level
# - Works in bull/bear: ADX filter ensures we only trade strong trends, Elder Ray captures momentum

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
    
    # Pre-compute 1d EMA(14) and ADX(14) for trend filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # EMA(14) for ADX calculation
    ema_14 = pd.Series(close_1d).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # True Range components
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_1d[0] = tr1[0]
    atr_1d = pd.Series(tr_1d).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # +DI and -DI calculation
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    up_move[0] = 0
    down_move[0] = 0
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    plus_di_1d = 100 * pd.Series(plus_dm).ewm(span=14, min_periods=14, adjust=False).mean().values / atr_1d
    minus_di_1d = 100 * pd.Series(minus_dm).ewm(span=14, min_periods=14, adjust=False).mean().values / atr_1d
    
    # ADX calculation
    dx = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d + 1e-10)
    adx_1d = pd.Series(dx).ewm(span=14, min_periods=14, adjust=False).mean().values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Pre-compute 6h EMA(13) for Elder Ray
    close_6h = prices['close'].values
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    volume_6h = prices['volume'].values
    
    ema_13_6h = pd.Series(close_6h).ewm(span=13, min_periods=13, adjust=False).mean().values
    
    # Elder Ray components
    bull_power = high_6h - ema_13_6h
    bear_power = ema_13_6h - low_6h
    
    # Pre-compute 6h volume filter
    avg_volume_20 = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_6h > (1.3 * avg_volume_20)
    
    # Pre-compute 6h ATR(14) for stoploss
    tr1_6h = high_6h - low_6h
    tr2_6h = np.abs(high_6h - np.roll(close_6h, 1))
    tr3_6h = np.abs(low_6h - np.roll(close_6h, 1))
    tr_6h = np.maximum(tr1_6h, np.maximum(tr2_6h, tr3_6h))
    tr_6h[0] = tr1_6h[0]
    atr_14_6h = pd.Series(tr_6h).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(ema_13_6h[i]) or np.isnan(bull_power[i]) or
            np.isnan(bear_power[i]) or np.isnan(vol_spike[i]) or np.isnan(atr_14_6h[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Bear Power becomes positive (bulls losing) OR stoploss hit
            if bear_power[i] > 0 or close_6h[i] < entry_price - 2.0 * atr_14_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Bull Power becomes positive (bears losing) OR stoploss hit
            if bull_power[i] > 0 or close_6h[i] > entry_price + 2.0 * atr_14_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Elder Ray signals with ADX and volume filters
            if adx_1d_aligned[i] > 25 and vol_spike[i]:
                # Long: Bull Power > 0 AND Bear Power < 0 (clear bullish momentum)
                if bull_power[i] > 0 and bear_power[i] < 0:
                    position = 1
                    entry_price = close_6h[i]
                    signals[i] = 0.25
                # Short: Bear Power > 0 AND Bull Power < 0 (clear bearish momentum)
                elif bear_power[i] > 0 and bull_power[i] < 0:
                    position = -1
                    entry_price = close_6h[i]
                    signals[i] = -0.25
    
    return signals