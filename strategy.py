#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour ADX(14) + PSAR trend following with volume confirmation
# Long when ADX > 25, PSAR below price (uptrend), and volume > 1.5x 20-period average
# Short when ADX > 25, PSAR above price (downtrend), and volume > 1.5x 20-period average
# Exit when ADX < 20 (trend weakening) or PSAR flips
# ADX filters weak trends, PSAR provides clear entry/exit, volume confirms strength
# Target: 75-200 total trades over 4 years (19-50/year) to balance opportunity and fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data for ADX and PSAR calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate ADX (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Directional Movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    atr = pd.Series(tr).ewm(span=14, adjust=False).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False).mean().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(span=14, adjust=False).mean().values
    
    # Parabolic SAR
    # Initialize
    psar = np.zeros_like(close_1d)
    psar[0] = low_1d[0]
    bull = True  # Start with bullish assumption
    af = 0.02  # Acceleration factor
    max_af = 0.2
    ep = high_1d[0] if bull else low_1d[0]  # Extreme point
    
    for i in range(1, len(close_1d)):
        if bull:
            psar[i] = psar[i-1] + af * (ep - psar[i-1])
            # Ensure SAR doesn't penetrate previous lows
            psar[i] = min(psar[i], low_1d[i-1], low_1d[i-2] if i >= 2 else low_1d[i-1])
            # Reverse if price breaks below SAR
            if low_1d[i] < psar[i]:
                bull = False
                psar[i] = ep
                af = 0.02
                ep = low_1d[i]
            else:
                # Continue bullish
                if high_1d[i] > ep:
                    ep = high_1d[i]
                    af = min(af + 0.02, max_af)
        else:
            psar[i] = psar[i-1] + af * (ep - psar[i-1])
            # Ensure SAR doesn't penetrate previous highs
            psar[i] = max(psar[i], high_1d[i-1], high_1d[i-2] if i >= 2 else high_1d[i-1])
            # Reverse if price breaks above SAR
            if high_1d[i] > psar[i]:
                bull = True
                psar[i] = ep
                af = 0.02
                ep = high_1d[i]
            else:
                # Continue bearish
                if low_1d[i] < ep:
                    ep = low_1d[i]
                    af = min(af + 0.02, max_af)
    
    # Calculate 20-period volume average
    vol_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align indicators to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    psar_aligned = align_htf_to_ltf(prices, df_1d, psar)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 30  # for ADX calculation
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(adx_aligned[i]) or np.isnan(psar_aligned[i]) or 
            np.isnan(vol_ma_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_1d_current = volume[i]  # Current 4h volume (using as proxy for 1d volume scaled)
        
        if position == 0:
            # Long setup: ADX > 25 (strong trend), PSAR below price (uptrend), volume confirmation
            if (adx_aligned[i] > 25 and 
                psar_aligned[i] < price and 
                vol_1d_current > 1.5 * vol_ma_20_aligned[i]):
                position = 1
                signals[i] = position_size
            # Short setup: ADX > 25 (strong trend), PSAR above price (downtrend), volume confirmation
            elif (adx_aligned[i] > 25 and 
                  psar_aligned[i] > price and 
                  vol_1d_current > 1.5 * vol_ma_20_aligned[i]):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: ADX < 20 (weakening trend) or PSAR flips above price
            if adx_aligned[i] < 20 or psar_aligned[i] > price:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: ADX < 20 (weakening trend) or PSAR flips below price
            if adx_aligned[i] < 20 or psar_aligned[i] < price:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_ADX_PSAR_Volume_Trend"
timeframe = "4h"
leverage = 1.0