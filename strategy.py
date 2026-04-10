#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray with 1d ADX regime and volume confirmation
# - Elder Ray measures institutional buying/selling pressure via High/Low vs EMA(13)
# - Long when Bull Power > 0 (buying pressure) in strong 1d uptrend (ADX > 25) with volume spike
# - Short when Bear Power < 0 (selling pressure) in strong 1d downtrend (ADX > 25) with volume spike
# - Exit when Elder Power crosses zero (pressure reversal)
# - Targets 12-37 trades/year (50-150 over 4 years) with discrete 0.25 position sizing
# - Works in bull/bear: ADX regime filters chop, volume confirms institutional participation

name = "6h_1d_elderray_adx_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute arrays
    open_6h = prices['open'].values
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    volume_6h = prices['volume'].values
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 6h EMA(13) for Elder Ray
    ema_13_6h = pd.Series(close_6h).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high_6h - ema_13_6h  # Buying pressure
    bear_power = low_6h - ema_13_6h   # Selling pressure
    
    # 1d ADX(14)
    tr1 = pd.Series(high_1d).diff()
    tr2 = abs(pd.Series(high_1d) - pd.Series(close_1d).shift(1))
    tr3 = abs(pd.Series(low_1d) - pd.Series(close_1d).shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = tr.rolling(window=14, min_periods=14).mean().values
    
    up_move = pd.Series(high_1d).diff()
    down_move = -pd.Series(low_1d).diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    plus_di = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr_1d
    minus_di = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr_1d
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx_1d = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    adx_1d = np.where((plus_di + minus_di) == 0, 0, adx_1d)
    
    # Align 1d indicators
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ma_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(50, n):
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(volume_ma_aligned[i]) or
            np.isnan(volume_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        adx_now = adx_aligned[i]
        vol_ma_now = volume_ma_aligned[i]
        vol_now = volume_1d_aligned[i]
        strong_trend = adx_now > 25
        volume_spike = vol_now > 1.5 * vol_ma_now
        
        if position == 0:
            if bull_power[i] > 0 and strong_trend and volume_spike:
                position = 1
                signals[i] = 0.25
            elif bear_power[i] < 0 and strong_trend and volume_spike:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:
            if position == 1:
                if bull_power[i] <= 0:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:
                if bear_power[i] >= 0:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals