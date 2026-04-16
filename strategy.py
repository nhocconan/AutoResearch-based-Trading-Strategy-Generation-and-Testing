#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R1/S1 breakout with 1d volume spike and 1w trend filter
# Uses 6h primary timeframe with 1d HTF for volume confirmation and 1w HTF for trend direction.
# Camarilla pivot levels identify intraday support/resistance; breakouts at R1/S1 with volume
# capture institutional participation. Weekly trend filter ensures alignment with higher timeframe
# momentum, reducing false breakouts in counter-trend periods. Works in both bull and bear
# markets by only taking breakouts aligned with weekly trend.
# Target: 50-150 trades over 4 years (12-37/year) to minimize fee drag while maintaining statistical significance.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 6h data (primary timeframe) ===
    df_6h = get_htf_data(prices, '6h')
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # === 1d data (HTF for Camarilla pivot calculation and volume confirmation) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # === 1w data (HTF for trend filter) ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # === 1d Camarilla pivot levels (based on previous day) ===
    # Calculate pivot using previous day's OHLC
    prev_close = np.roll(close_1d, 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    # First day will have NaN due to roll, handled by min_periods later
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_ = prev_high - prev_low
    r1 = pivot + (range_ * 1.1 / 12)
    s1 = pivot - (range_ * 1.1 / 12)
    r2 = pivot + (range_ * 1.1 / 6)
    s2 = pivot - (range_ * 1.1 / 6)
    r3 = pivot + (range_ * 1.1 / 4)
    s3 = pivot - (range_ * 1.1 / 4)
    r4 = pivot + (range_ * 1.1 / 2)
    s4 = pivot - (range_ * 1.1 / 2)
    
    # Align Camarilla levels to 6h timeframe (wait for 1d bar close)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # === 1d Volume confirmation ===
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_1d > (2.0 * vol_ma_20_1d)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike)
    
    # === 1w Trend filter (EMA34) ===
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 100
    
    # Track position and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(vol_spike_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or
            np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or
            np.isnan(s2_aligned[i]) or
            np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i]) or
            np.isnan(r4_aligned[i]) or
            np.isnan(s4_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        vol_conf = vol_spike_aligned[i]
        weekly_trend_up = ema_34_aligned[i]  # EMA value for trend comparison
        
        # Get weekly trend direction: price > EMA34 = uptrend
        # Need previous weekly close to determine trend
        if i >= 1:
            weekly_close_prev = close_1w[np.searchsorted(df_1w['open_time'].values, prices['open_time'].iloc[i]) - 1] if hasattr(df_1w['open_time'], 'values') else close_1w[max(0, (i // 28) - 1)]  # Approximate for safety
            # Simpler: use aligned EMA and price relationship
            weekly_close_aligned = align_htf_to_ltf(prices, df_1w, close_1w)
            weekly_close_val = weekly_close_aligned[i]
            weekly_trend_up = weekly_close_val > ema_34_aligned[i]
        else:
            weekly_trend_up = True  # default
        
        # === STOPLOSS LOGIC (ATR-based) ===
        if position == 1:  # Long position
            atr_6h = np.abs(high_6h - low_6h)
            atr_ma = pd.Series(atr_6h).rolling(window=14, min_periods=14).mean().values
            atr_aligned = align_htf_to_ltf(prices, df_6h, atr_ma)
            atr_val = atr_aligned[i]
            if price < entry_price - 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        elif position == -1:  # Short position
            atr_6h = np.abs(high_6h - low_6h)
            atr_ma = pd.Series(atr_6h).rolling(window=14, min_periods=14).mean().values
            atr_aligned = align_htf_to_ltf(prices, df_6h, atr_ma)
            atr_val = atr_aligned[i]
            if price > entry_price + 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when price reaches S1 (mean reversion) or shows weakness
            if price <= s1_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        elif position == -1:  # Short position
            # Exit when price reaches R1 (mean reversion) or shows strength
            if price >= r1_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Require volume spike
            if vol_conf:
                # Go long when price breaks above R1 with volume AND weekly uptrend
                if price > r1_aligned[i] and weekly_trend_up:
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
                    continue
                # Go short when price breaks below S1 with volume AND weekly downtrend
                elif price < s1_aligned[i] and not weekly_trend_up:
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

name = "6h_Camarilla_R1_S1_Breakout_1dVolumeSpike_1wTrendFilter"
timeframe = "6h"
leverage = 1.0