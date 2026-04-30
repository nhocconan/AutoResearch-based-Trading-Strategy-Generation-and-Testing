#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h TRIX(9) zero-cross + volume spike >1.8x + 1d ADX>25 trend filter
# TRIX catches momentum reversals with less whipsaw than MACD. Volume confirms institutional participation.
# 1d ADX>25 ensures we only trade in trending markets (avoids chop). Discrete sizing 0.25 limits fee drag.
# Works in bull/bear: TRIX zero-cross catches new trends, volume filter ensures legitimacy, ADX avoids false signals in ranging markets.

name = "4h_TRIX_ZeroCross_VolumeSpike_1dADX25_Trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate ATR for volatility (14-period)
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr2.iloc[0] = 0
    tr3.iloc[0] = 0
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # Calculate TRIX(9) on close
    ema1 = pd.Series(close).ewm(span=9, adjust=False, min_periods=9).mean()
    ema2 = ema1.ewm(span=9, adjust=False, min_periods=9).mean()
    ema3 = ema2.ewm(span=9, adjust=False, min_periods=9).mean()
    trix = 100 * (ema3 / ema3.shift(1) - 1)
    trix = trix.replace([np.inf, -np.inf], np.nan).fillna(0).values
    
    # Calculate 1d ADX(14) for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # ADX calculation
    plus_dm = pd.Series(df_1d['high']).diff()
    minus_dm = pd.Series(df_1d['low']).diff().mul(-1)
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    
    tr1 = pd.Series(df_1d['high'] - df_1d['low'])
    tr2 = pd.Series(np.abs(df_1d['high'] - df_1d['close'].shift(1)))
    tr3 = pd.Series(np.abs(df_1d['low'] - df_1d['close'].shift(1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr_1d = tr.rolling(window=14, min_periods=14).mean()
    plus_di_1d = 100 * (plus_dm.rolling(window=14, min_periods=14).sum() / atr_1d)
    minus_di_1d = 100 * (minus_dm.rolling(window=14, min_periods=14).sum() / atr_1d)
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)
    adx_1d = dx_1d.rolling(window=14, min_periods=14).mean().values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Volume confirmation: volume > 1.8x 30-period average
    vol_ma_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_confirm = volume > (1.8 * vol_ma_30)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(60, 30, 14)  # warmup for TRIX and ADX
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(trix[i]) or 
            np.isnan(adx_1d_aligned[i]) or 
            np.isnan(vol_ma_30[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
            
        curr_trix = trix[i]
        prev_trix = trix[i-1]
        curr_adx = adx_1d_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        curr_close = close[i]
        
        # TRIX zero-cross: bullish when crosses above 0, bearish when crosses below 0
        trix_bullish_cross = prev_trix <= 0 and curr_trix > 0
        trix_bearish_cross = prev_trix >= 0 and curr_trix < 0
        
        # Only trade in trending markets (ADX > 25) with volume confirmation
        if curr_adx > 25 and curr_volume_confirm:
            if position == 0:  # Flat - look for new entries
                if trix_bullish_cross:
                    signals[i] = 0.25
                    position = 1
                elif trix_bearish_cross:
                    signals[i] = -0.25
                    position = -1
            elif position == 1:  # Long position
                # Exit: TRIX crosses below zero
                if trix_bearish_cross:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:  # Short position
                # Exit: TRIX crosses above zero
                if trix_bullish_cross:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
        else:
            # In choppy or low volume conditions, stay flat
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
    
    return signals