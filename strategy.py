#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Bollinger Band squeeze breakout with 1d volume confirmation and RSI filter.
# Uses BB width percentile to detect low volatility squeezes (breakout precursors).
# In squeeze: wait for breakout above/below BB with volume > 2x average and RSI confirming momentum.
# Position size 0.25 to manage drawdown. Designed to capture explosive moves in both bull and bear markets.

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
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    volume_4h = df_4h['volume'].values
    
    # === 1d data (higher timeframe for volume confirmation) ===
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    
    # === 4h Bollinger Bands (20, 2) ===
    bb_middle = pd.Series(close_4h).rolling(window=20, min_periods=20).mean().values
    bb_std = pd.Series(close_4h).rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    bb_width = bb_upper - bb_lower
    
    # === 4h BB width percentile (50) for squeeze detection ===
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=50, min_periods=50).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    
    # === 4h RSI(14) for momentum confirmation ===
    delta = pd.Series(close_4h).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.fillna(50).values
    
    # === 4h volume ratio for breakout confirmation ===
    vol_ma_20_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    vol_ratio_4h = volume_4h / vol_ma_20_4h
    
    # === 1d volume ratio for higher timeframe confirmation ===
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ratio_1d = volume_1d / vol_ma_20_1d
    
    # Align HTF indicators to 4h timeframe
    bb_width_percentile_aligned = align_htf_to_ltf(prices, df_4h, bb_width_percentile)
    rsi_aligned = align_htf_to_ltf(prices, df_4h, rsi_values)
    vol_ratio_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ratio_4h)
    vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 100
    
    # Track position and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(bb_width_percentile_aligned[i]) or 
            np.isnan(rsi_aligned[i]) or
            np.isnan(vol_ratio_4h_aligned[i]) or
            np.isnan(vol_ratio_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        bb_width_pct = bb_width_percentile_aligned[i]
        rsi_val = rsi_aligned[i]
        vol_ratio_4h = vol_ratio_4h_aligned[i]
        vol_ratio_1d = vol_ratio_1d_aligned[i]
        
        # === STOPLOSS LOGIC ===
        if position == 1:  # Long position
            # Stop loss: price closes below entry - 2.0 * ATR (using BB width as proxy)
            bb_width_val = bb_width[i]
            if price < entry_price - 2.0 * bb_width_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        elif position == -1:  # Short position
            # Stop loss: price closes above entry + 2.0 * ATR (using BB width as proxy)
            bb_width_val = bb_width[i]
            if price > entry_price + 2.0 * bb_width_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit: RSI overbought or BB width expansion (end of squeeze)
            if rsi_val > 70 or bb_width_pct > 80:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        elif position == -1:  # Short position
            # Exit: RSI oversold or BB width expansion (end of squeeze)
            if rsi_val < 30 or bb_width_pct > 80:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Squeeze condition: BB width in lowest 20th percentile
            if bb_width_pct < 20:
                # LONG: breakout above upper BB with volume and RSI confirmation
                if (price > bb_upper[i] and 
                    vol_ratio_4h > 2.0 and 
                    vol_ratio_1d > 1.5 and 
                    rsi_val > 50):
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
                    continue
                # SHORT: breakout below lower BB with volume and RSI confirmation
                elif (price < bb_lower[i] and 
                      vol_ratio_4h > 2.0 and 
                      vol_ratio_1d > 1.5 and 
                      rsi_val < 50):
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

name = "4h_BB_Squeeze_Breakout_Volume_RSI_v1"
timeframe = "4h"
leverage = 1.0