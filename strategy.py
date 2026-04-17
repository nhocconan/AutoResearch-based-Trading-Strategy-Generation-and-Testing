# 12h DMI Trend + Parabolic SAR + Volume Filter
# Hypothesis: Trend-following using DMI (ADX) for trend strength and Parabolic SAR for entries, filtered by volume.
# Works in both bull and bear by only taking strong trends (ADX > 25) and using SAR for precise entry/exit.
# Target: 20-30 trades/year per symbol.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1D data for DMI and SAR
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate DMI (ADX) on 1D
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # Directional Movement
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Smoothed values
    tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    plus_dm14 = pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values
    minus_dm14 = pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values
    
    # DI and DX
    plus_di14 = 100 * plus_dm14 / tr14
    minus_di14 = 100 * minus_dm14 / tr14
    dx = 100 * np.abs(plus_di14 - minus_di14) / (plus_di14 + minus_di14)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Parabolic SAR
    def calculate_psar(high, low, af_start=0.02, af_increment=0.02, af_max=0.2):
        psar = np.zeros_like(high)
        psar[0] = low[0]
        trend = 1  # 1 for up, -1 for down
        af = af_start
        ep = high[0] if trend == 1 else low[0]
        
        for i in range(1, len(high)):
            if trend == 1:
                psar[i] = psar[i-1] + af * (ep - psar[i-1])
                if low[i] < psar[i]:
                    trend = -1
                    psar[i] = ep
                    af = af_start
                    ep = low[i]
                else:
                    if high[i] > ep:
                        ep = high[i]
                        af = min(af + af_increment, af_max)
            else:
                psar[i] = psar[i-1] + af * (ep - psar[i-1])
                if high[i] > psar[i]:
                    trend = 1
                    psar[i] = ep
                    af = af_start
                    ep = high[i]
                else:
                    if low[i] < ep:
                        ep = low[i]
                        af = min(af + af_increment, af_max)
        return psar
    
    psar = calculate_psar(high_1d, low_1d)
    
    # Align to 12h
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    psar_aligned = align_htf_to_ltf(prices, df_1d, psar)
    
    # Volume filter: 12h volume > 1.5x 20-period MA
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(adx_aligned[i]) or np.isnan(psar_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        # Trend filter: ADX > 25
        if adx_aligned[i] <= 25:
            # No strong trend, stay flat
            signals[i] = 0.0
            position = 0
            continue
        
        if position == 0:
            # Long: price > SAR and volume filter
            if price > psar_aligned[i] and vol > 1.5 * vol_ma[i]:
                signals[i] = 0.25
                position = 1
            # Short: price < SAR and volume filter
            elif price < psar_aligned[i] and vol > 1.5 * vol_ma[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: exit when price < SAR
            if price < psar_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short: exit when price > SAR
            if price > psar_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_DMI_PSAR_VolumeFilter"
timeframe = "12h"
leverage = 1.0