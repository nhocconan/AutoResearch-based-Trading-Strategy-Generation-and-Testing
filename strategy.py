#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h TRIX momentum with 1d volume regime and ATR stoploss
# Uses 4h primary timeframe with 1d HTF for volume regime detection (expanding volume = momentum phase).
# TRIX (12) captures smoothed momentum with reduced whipsaw vs raw MACD.
# Volume regime: 1d volume > 1.5x 20-period average confirms institutional participation.
# ATR-based stoploss (2.5x) and Donchian(20) exit for risk management.
# Target: 75-200 total trades over 4 years (19-50/year) to balance statistical significance and fee drag.
# Works in bull markets via long TRIX crossovers and in bear markets via short signals during volume expansion.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4h data (primary timeframe) ===
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # === 1d data (HTF for volume regime) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # === 4h TRIX (12,9,9) - triple smoothed EMA of ROC ===
    # ROC(12) = (close/close.shift(12) - 1) * 100
    roc = np.zeros_like(close_4h)
    roc[12:] = (close_4h[12:] / close_4h[:-12] - 1) * 100
    # EMA1 of ROC
    ema1 = pd.Series(roc).ewm(span=12, adjust=False, min_periods=12).mean().values
    # EMA2 of EMA1
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean().values
    # EMA3 of EMA2 = TRIX
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean().values
    trix = ema3
    trix_signal = pd.Series(trix).ewm(span=9, adjust=False, min_periods=9).mean().values
    trix_hist = trix - trix_signal  # MACD-style histogram
    
    # Align TRIX histogram to 4h timeframe (wait for 4h bar close)
    trix_hist_aligned = align_htf_to_ltf(prices, df_4h, trix_hist)
    
    # === 1d Volume regime filter (expanding volume) ===
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_regime = volume_1d > (1.5 * vol_ma_20_1d)  # True when volume expanding
    vol_regime_aligned = align_htf_to_ltf(prices, df_1d, vol_regime)
    
    # === 4h Donchian channels (20-period) for exit ===
    donch_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donch_high_aligned = align_htf_to_ltf(prices, df_4h, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_4h, donch_low)
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 100
    
    # Track position and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(trix_hist_aligned[i]) or 
            np.isnan(vol_regime_aligned[i]) or
            np.isnan(donch_high_aligned[i]) or
            np.isnan(donch_low_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        regime_ok = vol_regime_aligned[i]
        
        # === STOPLOSS LOGIC (ATR-based) ===
        if position == 1:  # Long position
            atr_4h = np.abs(high_4h - low_4h)
            atr_ma = pd.Series(atr_4h).ewm(span=14, adjust=False, min_periods=14).mean().values
            atr_aligned = align_htf_to_ltf(prices, df_4h, atr_ma)
            atr_val = atr_aligned[i]
            if price < entry_price - 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        elif position == -1:  # Short position
            atr_4h = np.abs(high_4h - low_4h)
            atr_ma = pd.Series(atr_4h).ewm(span=14, adjust=False, min_periods=14).mean().values
            atr_aligned = align_htf_to_ltf(prices, df_4h, atr_ma)
            atr_val = atr_aligned[i]
            if price > entry_price + 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # === EXIT LOGIC (Donchian breakout in opposite direction) ===
        if position == 1:  # Long position
            # Exit when price breaks below Donchian low
            if price < donch_low_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        elif position == -1:  # Short position
            # Exit when price breaks above Donchian high
            if price > donch_high_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Require volume regime (expanding volume)
            if regime_ok:
                # Go long when TRIX histogram crosses above zero (bullish momentum)
                if trix_hist_aligned[i] > 0 and trix_hist_aligned[i-1] <= 0:
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
                    continue
                # Go short when TRIX histogram crosses below zero (bearish momentum)
                elif trix_hist_aligned[i] < 0 and trix_hist_aligned[i-1] >= 0:
                    signals[i] = -0.25
                    position = -1
                    entry_price = price
                    continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_TRIX12_VolumeRegime_ATRStop"
timeframe = "4h"
leverage = 1.0