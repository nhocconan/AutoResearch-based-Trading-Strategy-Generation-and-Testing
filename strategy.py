#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_Volume_Regime_ATRStop_V1
Hypothesis: 4h Camarilla R1/S1 breakout with volume confirmation (>1.4x 20-period volume MA) and regime filter (ADX > 25 for trending, ADX < 20 for choppy). Uses 1d HTF EMA50 for trend bias (price > EMA50 for longs, < EMA50 for shorts). ATR-based stoploss via signal=0 when price moves against position by 2.0*ATR. Designed for low trade frequency (<200 total 4h trades) to minimize fee drag and work in both bull/bear markets via regime adaptation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for EMA trend filter)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d EMA50 for trend filter ===
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === 4h Indicators (primary timeframe) ===
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Camarilla pivot levels (based on previous day's OHLC)
    # For intraday, we use previous 4h bar's OHLC as proxy for daily
    # But since we're on 4h timeframe, we calculate camarilla from previous day's data
    # We'll approximate using rolling window of 6 bars (1.5 days) for simplicity
    # Proper camarilla needs daily OHLC, so we use 1d data resampled conceptually
    # Instead, we use the 1d OHLC from HTF data
    
    # Calculate camarilla levels from 1d OHLC (proper method)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    open_1d = df_1d['open'].values  # assuming open column exists
    
    camarilla_h4 = []
    camarilla_l4 = []
    camarilla_h3 = []
    camarilla_l3 = []
    camarilla_h2 = []
    camarilla_l2 = []
    camarilla_h1 = []
    camarilla_l1 = []
    camarilla_p = []
    
    for i in range(len(close_1d)):
        if i == 0:
            camarilla_h4.append(np.nan)
            camarilla_l4.append(np.nan)
            camarilla_h3.append(np.nan)
            camarilla_l3.append(np.nan)
            camarilla_h2.append(np.nan)
            camarilla_l2.append(np.nan)
            camarilla_h1.append(np.nan)
            camarilla_l1.append(np.nan)
            camarilla_p.append(np.nan)
        else:
            # Previous day's OHLC
            ph = high_1d[i-1]
            pl = low_1d[i-1]
            pc = close_1d[i-1]
            po = open_1d[i-1]
            
            # Typical price for pivot
            pivot = (ph + pl + pc) / 3
            range_ = ph - pl
            
            # Camarilla levels
            h4 = pc + (range_ * 1.1 / 2)
            l4 = pc - (range_ * 1.1 / 2)
            h3 = pc + (range_ * 1.1 / 4)
            l3 = pc - (range_ * 1.1 / 4)
            h2 = pc + (range_ * 1.1 / 6)
            l2 = pc - (range_ * 1.1 / 6)
            h1 = pc + (range_ * 1.1 / 12)
            l1 = pc - (range_ * 1.1 / 12)
            
            camarilla_h4.append(h4)
            camarilla_l4.append(l4)
            camarilla_h3.append(h3)
            camarilla_l3.append(l3)
            camarilla_h2.append(h2)
            camarilla_l2.append(l2)
            camarilla_h1.append(h1)
            camarilla_l1.append(l1)
            camarilla_p.append(pivot)
    
    camarilla_h4 = np.array(camarilla_h4)
    camarilla_l4 = np.array(camarilla_l4)
    camarilla_h3 = np.array(camarilla_h3)
    camarilla_l3 = np.array(camarilla_l3)
    camarilla_h2 = np.array(camarilla_h2)
    camarilla_l2 = np.array(camarilla_l2)
    camarilla_h1 = np.array(camarilla_h1)
    camarilla_l1 = np.array(camarilla_l1)
    camarilla_p = np.array(camarilla_p)
    
    # Align camarilla levels to 4h timeframe
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_h2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h2)
    camarilla_l2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l2)
    camarilla_h1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h1)
    camarilla_l1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l1)
    camarilla_p_aligned = align_htf_to_ltf(prices, df_1d, camarilla_p)
    
    # Volume MA (20-period) for spike detection
    vol_ma = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    # ATR (14-period) for stoploss
    tr1 = pd.Series(high_4h - low_4h)
    tr2 = pd.Series(np.abs(high_4h - np.roll(close_4h, 1)))
    tr3 = pd.Series(np.abs(low_4h - np.roll(close_4h, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # ADX (14-period) for regime filter
    plus_dm = pd.Series(np.where((high_4h - np.roll(high_4h, 1)) > (np.roll(low_4h, 1) - low_4h),
                                 np.maximum(high_4h - np.roll(high_4h, 1), 0), 0))
    minus_dm = pd.Series(np.where((np.roll(low_4h, 1) - low_4h) > (high_4h - np.roll(high_4h, 1)),
                                  np.maximum(np.roll(low_4h, 1) - low_4h, 0), 0))
    tr_14 = tr.rolling(window=14, min_periods=1).sum()
    plus_di_14 = 100 * (plus_dm.rolling(window=14, min_periods=14).sum() / tr_14)
    minus_di_14 = 100 * (minus_dm.rolling(window=14, min_periods=14).sum() / tr_14)
    dx = 100 * np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14)
    adx = dx.rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(camarilla_h1_aligned[i]) or np.isnan(camarilla_l1_aligned[i]) 
            or np.isnan(vol_ma[i]) or np.isnan(atr[i]) or np.isnan(adx[i])
            or np.isnan(ema_50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_4h[i]
        vol = volume_4h[i]
        vol_ok = vol > 1.4 * vol_ma[i]  # volume confirmation
        
        # Regime detection
        is_trending = adx[i] > 25  # trending regime
        is_choppy = adx[i] < 20  # choppy regime
        
        if position == 0:
            # Long: Camarilla H1 breakout + volume + trend bias
            if price > camarilla_h1_aligned[i] and vol_ok and (price > ema_50_1d_aligned[i] or is_choppy):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: Camarilla L1 breakdown + volume + trend bias
            elif price < camarilla_l1_aligned[i] and vol_ok and (price < ema_50_1d_aligned[i] or is_choppy):
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Check stoploss
            if price < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Exit conditions: Camarilla L1 break or loss of volume/momentum
            elif price < camarilla_l1_aligned[i] or not vol_ok:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Check stoploss
            if price > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Exit conditions: Camarilla H1 break or loss of volume/momentum
            elif price > camarilla_h1_aligned[i] or not vol_ok:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_Volume_Regime_ATRStop_V1"
timeframe = "4h"
leverage = 1.0