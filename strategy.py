#!/usr/bin/env python3
"""
1d_1w_Three_Level_System
Hypothesis: Use 1w ATR regime to switch between mean reversion (low vol) and trend following (high vol).
In low volatility (1w ATR < 40th percentile): mean revert at 1d Bollinger Bands with RSI filter.
In high volatility: trade 1d Donchian breakouts with volume confirmation.
Designed for BTC/ETH to work in both bull and bear by adapting to market regime.
Timeframe: 1d, HTF: 1w for regime detection.
Target: 15-30 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_Three_Level_System"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === WEEKLY DATA FOR REGIME DETECTION ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly ATR(14) for regime detection
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr_1w = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1w = pd.Series(tr_1w).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Weekly ATR percentile (40-period lookback) - regime filter: <0.4 = low vol
    atr_series = pd.Series(atr_1w)
    atr_percentile = atr_series.rolling(window=40, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5, raw=False
    ).values
    atr_regime = align_htf_to_ltf(prices, df_1w, atr_percentile)  # < 0.4 = low vol regime
    
    # === DAILY DATA FOR INDICATORS ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Bollinger Bands for mean reversion (low vol regime)
    sma_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    bb_upper = sma_20 + 2.0 * std_20
    bb_lower = sma_20 - 2.0 * std_20
    
    # RSI for mean reversion filter
    delta = pd.Series(close_1d).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Donchian Channel for trend following (high vol regime)
    donch_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    
    # Align all daily indicators
    sma_20_aligned = align_htf_to_ltf(prices, df_1d, sma_20)
    bb_upper_aligned = align_htf_to_ltf(prices, df_1d, bb_upper)
    bb_lower_aligned = align_htf_to_ltf(prices, df_1d, bb_lower)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low)
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma)
    vol_ratio_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Need enough lookback
        # Skip if not ready
        if (np.isnan(atr_regime[i]) or np.isnan(sma_20_aligned[i]) or 
            np.isnan(bb_upper_aligned[i]) or np.isnan(bb_lower_aligned[i]) or
            np.isnan(rsi_aligned[i]) or np.isnan(donch_high_aligned[i]) or
            np.isnan(donch_low_aligned[i]) or np.isnan(vol_ratio_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Regime detection
        low_vol_regime = atr_regime[i] < 0.4  # Low volatility = mean revert
        high_vol_regime = atr_regime[i] >= 0.4  # High volatility = trend follow
        
        if low_vol_regime:
            # LOW VOL: Mean reversion at Bollinger Bands with RSI filter
            long_signal = (close[i] <= bb_lower_aligned[i]) and (rsi_aligned[i] < 30)
            short_signal = (close[i] >= bb_upper_aligned[i]) and (rsi_aligned[i] > 70)
            
            # Exit when price returns to SMA or RSI normalizes
            exit_long = (close[i] >= sma_20_aligned[i]) or (rsi_aligned[i] >= 50)
            exit_short = (close[i] <= sma_20_aligned[i]) or (rsi_aligned[i] <= 50)
            
        else:  # high_vol_regime
            # HIGH VOL: Trend following with Donchian breakouts + volume
            long_signal = (close[i] > donch_high_aligned[i]) and (vol_ratio_aligned[i] > 1.5)
            short_signal = (close[i] < donch_low_aligned[i]) and (vol_ratio_aligned[i] > 1.5)
            
            # Exit when price returns to opposite Donchian band
            exit_long = close[i] <= donch_low_aligned[i]
            exit_short = close[i] >= donch_high_aligned[i]
        
        # Execute trades
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals