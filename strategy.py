#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Bollinger Band Squeeze + Volume Spike + RSI Reversion
# Uses Bollinger Bands width percentile to detect low volatility squeeze (breakout precursor).
# Enters long/short on price breaking BB ±2σ with volume > 1.5x average and RSI confirming momentum.
# Works in both bull and bear markets by capturing volatility expansion phases.
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
    
    # === Bollinger Bands (20, 2) on 4h close ===
    close_series = pd.Series(close_4h)
    bb_middle = close_series.rolling(window=20, min_periods=20).mean()
    bb_std = close_series.rolling(window=20, min_periods=20).std()
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    bb_width = bb_upper - bb_lower
    
    # Bollinger Band width percentile (50-day lookback) to detect squeeze
    bb_width_series = pd.Series(bb_width.values)
    bb_width_percentile = bb_width_series.rolling(window=50, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5, raw=False
    ).values
    squeeze_condition = bb_width_percentile < 0.2  # Bottom 20% = squeeze
    
    # === Breakout detection ===
    breakout_up = close_4h > bb_upper
    breakout_down = close_4h < bb_lower
    
    # === Volume spike detection ===
    vol_ma_20_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_4h > (1.5 * vol_ma_20_4h)
    
    # === RSI(14) for momentum confirmation ===
    delta = pd.Series(close_4h).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Align indicators to LTF
    squeeze_aligned = align_htf_to_ltf(prices, df_4h, squeeze_condition)
    breakout_up_aligned = align_htf_to_ltf(prices, df_4h, breakout_up)
    breakout_down_aligned = align_htf_to_ltf(prices, df_4h, breakout_down)
    vol_spike_aligned = align_htf_to_ltf(prices, df_4h, vol_spike)
    rsi_aligned = align_htf_to_ltf(prices, df_4h, rsi_values)
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 50
    
    # Track position and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(squeeze_aligned[i]) or 
            np.isnan(breakout_up_aligned[i]) or
            np.isnan(breakout_down_aligned[i]) or
            np.isnan(vol_spike_aligned[i]) or
            np.isnan(rsi_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        is_squeeze = squeeze_aligned[i] > 0.5
        is_breakout_up = breakout_up_aligned[i] > 0.5
        is_breakout_down = breakout_down_aligned[i] > 0.5
        vol_spike_val = vol_spike_aligned[i]
        rsi_val = rsi_aligned[i]
        
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
            # Exit when breakout fails or RSI overbought
            if not is_breakout_up or rsi_val > 70:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        elif position == -1:  # Short position
            # Exit when breakout fails or RSI oversold
            if not is_breakout_down or rsi_val < 30:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Require squeeze condition, volume spike, and breakout with RSI confirmation
            if is_squeeze and vol_spike_val:
                # Go long on upward breakout with RSI > 50 (bullish momentum)
                if is_breakout_up and rsi_val > 50:
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
                    continue
                # Go short on downward breakout with RSI < 50 (bearish momentum)
                elif is_breakout_down and rsi_val < 50:
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

name = "4h_BB_Squeeze_Volume_RSI_Breakout_v1"
timeframe = "4h"
leverage = 1.0