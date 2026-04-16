#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h KAMA trend filter and 1d Donchian breakout with volume confirmation.
# Long when 4h KAMA slope > 0 AND price breaks above 1d Donchian upper band (20) AND volume > 1.5x 20-period 4h average volume.
# Short when 4h KAMA slope < 0 AND price breaks below 1d Donchian lower band (20) AND volume > 1.5x 20-period 4h average volume.
# Exit when price crosses 1d Donchian midline or ATR-based stop (1.5*ATR from entry).
# Uses discrete position size 0.20. Designed to capture strong breakouts in trending markets with volume confirmation.
# Target: 60-150 total trades over 4 years (15-37/year) to balance opportunity and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4h Indicators: KAMA trend filter ===
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Efficiency Ratio (ER) for KAMA
    change = np.abs(np.diff(close_4h, prepend=close_4h[0]))
    volatility = np.sum(np.abs(np.diff(close_4h, prepend=close_4h[0])), axis=0)  # placeholder, will compute properly below
    
    # Proper ER calculation: |current - close N periods ago| / sum(|diff|) over N periods
    lookback = 10
    diff = np.diff(close_4h, prepend=close_4h[0])
    volatility = np.zeros_like(close_4h)
    for i in range(lookback, len(close_4h)):
        volatility[i] = np.sum(np.abs(diff[i-lookback+1:i+1]))
    price_change = np.abs(np.subtract(close_4h[lookback:], close_4h[:-lookback]))
    price_change = np.concatenate([np.full(lookback, np.nan), price_change])
    er = np.divide(price_change, volatility, out=np.full_like(price_change, np.nan), where=volatility!=0)
    er = np.nan_to_num(er, nan=0.0)
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)
    slow_sc = 2 / (30 + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.full_like(close_4h, np.nan)
    kama[lookback] = close_4h[lookback]  # seed
    for i in range(lookback + 1, len(close_4h)):
        kama[i] = kama[i-1] + sc[i] * (close_4h[i] - kama[i-1])
    
    # KAMA slope (trend direction)
    kama_slope = np.diff(kama, prepend=0)
    kama_slope_aligned = align_htf_to_ltf(prices, df_4h, kama_slope)
    
    # === 1d Indicators: Donchian Channel (20-period) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Donchian upper and lower bands (20-period)
    dc_upper_1d = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    dc_lower_1d = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    dc_mid_1d = (dc_upper_1d + dc_lower_1d) / 2
    
    # Align to 1h timeframe
    dc_upper_1d_aligned = align_htf_to_ltf(prices, df_1d, dc_upper_1d)
    dc_lower_1d_aligned = align_htf_to_ltf(prices, df_1d, dc_lower_1d)
    dc_mid_1d_aligned = align_htf_to_ltf(prices, df_1d, dc_mid_1d)
    
    # === 4h Indicators: Volume Spike (volume > 1.5x 20-period average) ===
    vol_4h = df_4h['volume'].values
    vol_ma_4h = pd.Series(vol_4h).rolling(window=20, min_periods=20).mean().values
    vol_ma_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_4h)
    volume_spike = volume > (1.5 * vol_ma_4h_aligned)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    # Track position state and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Calculate 1h ATR for stoploss
    tr1 = pd.Series(high).diff()
    tr2 = pd.Series(low).diff().abs()
    tr3 = pd.Series(close).shift(1).diff().abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_raw = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(kama_slope_aligned[i]) or np.isnan(dc_upper_1d_aligned[i]) or np.isnan(dc_lower_1d_aligned[i]) or
            np.isnan(dc_mid_1d_aligned[i]) or np.isnan(volume_spike[i]) or np.isnan(atr_raw[i]) or
            not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        kama_slope_val = kama_slope_aligned[i]
        vol_spike = volume_spike[i]
        atr_val = atr_raw[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price crosses below midline
            if price < dc_mid_1d_aligned[i]:
                exit_signal = True
            # ATR-based stoploss: 1.5*ATR below entry
            elif price < entry_price - 1.5 * atr_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price crosses above midline
            if price > dc_mid_1d_aligned[i]:
                exit_signal = True
            # ATR-based stoploss: 1.5*ATR above entry
            elif price > entry_price + 1.5 * atr_val:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: 4h KAMA slope > 0 AND price breaks above 1d Donchian upper AND volume spike
            if kama_slope_val > 0 and price > dc_upper_1d_aligned[i] and vol_spike:
                signals[i] = 0.20
                position = 1
                entry_price = price
            
            # SHORT: 4h KAMA slope < 0 AND price breaks below 1d Donchian lower AND volume spike
            elif kama_slope_val < 0 and price < dc_lower_1d_aligned[i] and vol_spike:
                signals[i] = -0.20
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.20
    
    return signals

name = "1h_KAMA4hTrend_Donchian1d_VolumeSpike_V1"
timeframe = "1h"
leverage = 1.0