#!/usr/bin/env python3
"""
12h_Triple_Confluence_Strategy
Hypothesis: Combine Donchian breakout (trend), RSI mean-reversion (momentum), 
and volume confirmation with regime filtering (ADX) to capture high-probability 
trades in both bull and bear markets. Uses 12h for execution, 1d for trend filter.
Target: 20-40 trades/year with disciplined multi-factor entry.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 12h data (primary) ===
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    volume_12h = df_12h['volume'].values
    
    # === 1d data (HTF for trend filter) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # === Indicators on 12h timeframe ===
    # Donchian channels (20-period)
    donchian_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # RSI (14-period)
    delta = pd.Series(close_12h).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # Volume ratio (20-period average)
    vol_ma_20 = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume_12h / vol_ma_20
    
    # === ADX on 1d timeframe (regime filter) ===
    # Need high/low/close for 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Wilder's smoothing
    def wilders_smoothing(x, period):
        result = np.full_like(x, np.nan)
        if len(x) >= period:
            result[period-1] = np.nanmean(x[1:period])
            for i in range(period, len(x)):
                result[i] = result[i-1] - (result[i-1]/period) + x[i]
        return result
    
    atr_1d = wilders_smoothing(tr, 14)
    dm_plus_smooth = wilders_smoothing(dm_plus, 14)
    dm_minus_smooth = wilders_smoothing(dm_minus, 14)
    
    di_plus = np.where(atr_1d != 0, 100 * dm_plus_smooth / atr_1d, 0)
    di_minus = np.where(atr_1d != 0, 100 * dm_minus_smooth / atr_1d, 0)
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx_1d = wilders_smoothing(dx, 14)
    
    # === Align HTF data to 12h timeframe ===
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low)
    rsi_aligned = align_htf_to_ltf(prices, df_12h, rsi)
    vol_ratio_aligned = align_htf_to_ltf(prices, df_12h, vol_ratio)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 60
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or 
            np.isnan(rsi_aligned[i]) or 
            np.isnan(vol_ratio_aligned[i]) or 
            np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        upper = donchian_high_aligned[i]
        lower = donchian_low_aligned[i]
        rsi_val = rsi_aligned[i]
        vol_ratio_val = vol_ratio_aligned[i]
        adx_val = adx_aligned[i]
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit: price closes below Donchian low OR RSI overbought
            if price < lower or rsi_val > 70:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit: price closes above Donchian high OR RSI oversold
            if price > upper or rsi_val < 30:
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Require strong trend (ADX > 25) for breakout trades
            if adx_val > 25:
                # LONG: Break above upper band with volume and RSI not overbought
                if price > upper and vol_ratio_val > 1.3 and rsi_val < 70:
                    signals[i] = 0.25
                    position = 1
                    continue
                # SHORT: Break below lower band with volume and RSI not oversold
                elif price < lower and vol_ratio_val > 1.3 and rsi_val > 30:
                    signals[i] = -0.25
                    position = -1
                    continue
            # In weak trend (ADX <= 25), use mean reversion at extremes
            else:
                # LONG: Near lower band with oversold RSI and volume confirmation
                if price <= lower * 1.005 and rsi_val < 30 and vol_ratio_val > 1.2:
                    signals[i] = 0.25
                    position = 1
                    continue
                # SHORT: Near upper band with overbought RSI and volume confirmation
                elif price >= upper * 0.995 and rsi_val > 70 and vol_ratio_val > 1.2:
                    signals[i] = -0.25
                    position = -1
                    continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "12h_Triple_Confluence_Strategy"
timeframe = "12h"
leverage = 1.0