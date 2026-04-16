#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h ADX Trend + RSI Pullback Entry with Volume Confirmation
# Uses ADX(14) > 25 to identify trending markets, then enters on RSI(14) pullbacks
# (long when RSI < 40 in uptrend, short when RSI > 60 in downtrend) with volume > 1.5x average.
# Works in both bull and bear markets by following the trend direction.
# Target: 80-180 total trades over 4 years (20-45/year) with disciplined entries.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
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
    open_4h = df_4h['open'].values
    
    # === 1d data (higher timeframe for ADX trend filter) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # === 4x RSI(14) calculation ===
    delta = pd.Series(close_4h).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # === 4x volume spike detection ===
    vol_ma_20_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_4h > (1.5 * vol_ma_20_4h)
    
    # === 1d ADX(14) for trend filter ===
    # Calculate True Range
    tr1 = pd.Series(high_1d).diff()
    tr2 = abs(pd.Series(high_1d).diff())
    tr3 = abs(pd.Series(low_1d).diff())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = tr.rolling(window=14, min_periods=14).mean().values
    
    # Calculate Directional Movement
    up_move = pd.Series(high_1d).diff()
    down_move = -pd.Series(low_1d).diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smooth DM and TR
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / atr_1d
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / atr_1d
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Determine trend direction from +DI/-DI crossover
    plus_di_smooth = pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values
    minus_di_smooth = pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values
    di_diff = plus_di_smooth - minus_di_smooth
    # Uptrend when +DI > -DI, downtrend when -DI > +DI
    uptrend = di_diff > 0
    downtrend = di_diff < 0
    uptrend_aligned = align_htf_to_ltf(prices, df_1d, uptrend.astype(float))
    downtrend_aligned = align_htf_to_ltf(prices, df_1d, downtrend.astype(float))
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 50
    
    # Track position and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(adx_aligned[i]) or 
            np.isnan(rsi_values[i]) or
            np.isnan(vol_ma_20_4h[i]) or
            np.isnan(uptrend_aligned[i]) or
            np.isnan(downtrend_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        adx_val = adx_aligned[i]
        rsi_val = rsi_values[i]
        vol_spike_val = vol_spike[i]
        is_uptrend = uptrend_aligned[i] > 0.5
        is_downtrend = downtrend_aligned[i] > 0.5
        
        # === STOPLOSS LOGIC ===
        if position == 1:  # Long position
            atr_4h = np.abs(high_4h - low_4h)
            atr_ma = pd.Series(atr_4h).rolling(window=14, min_periods=14).mean().values
            atr_aligned = align_htf_to_ltf(prices, df_4h, atr_ma)
            atr_val = atr_aligned[i]
            if price < entry_price - 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        elif position == -1:  # Short position
            atr_4h = np.abs(high_4h - low_4h)
            atr_ma = pd.Series(atr_4h).rolling(window=14, min_periods=14).mean().values
            atr_aligned = align_htf_to_ltf(prices, df_4h, atr_ma)
            atr_val = atr_aligned[i]
            if price > entry_price + 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when trend weakens or RSI overbought
            if adx_val < 20 or rsi_val > 70:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        elif position == -1:  # Short position
            # Exit when trend weakens or RSI oversold
            if adx_val < 20 or rsi_val < 30:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Require trending market (ADX > 25) and volume spike
            if adx_val > 25 and vol_spike_val:
                # Go long in uptrend on RSI pullback (< 40)
                if is_uptrend and rsi_val < 40:
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
                    continue
                # Go short in downtrend on RSI bounce (> 60)
                elif is_downtrend and rsi_val > 60:
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

name = "4h_ADXTrend_RSIPullback_Volume_v1"
timeframe = "4h"
leverage = 1.0