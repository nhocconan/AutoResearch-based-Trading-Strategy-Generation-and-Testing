#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Bollinger Band width percentile + 1d RSI mean reversion + volume confirmation
# Long when BB width < 20th percentile AND RSI < 30 AND volume > 1.5x 12h average
# Short when BB width < 20th percentile AND RSI > 70 AND volume > 1.5x 12h average
# ATR trailing stop (1.5x ATR) to manage risk
# Bollinger squeeze identifies low volatility periods preceding breakouts, RSI captures overextended moves, volume confirms conviction
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 12h Bollinger Bands (20-period, 2 std dev) ===
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate Bollinger Bands
    ma_20 = pd.Series(close_12h).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_12h).rolling(window=20, min_periods=20).std().values
    upper_bb = ma_20 + 2 * std_20
    lower_bb = ma_20 - 2 * std_20
    bb_width = upper_bb - lower_bb
    
    # BB width percentile (20-period lookback)
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=20, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    bb_width_percentile_aligned = align_htf_to_ltf(prices, df_12h, bb_width_percentile)
    
    # === 1d RSI (14-period) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate RSI
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # === 12h Volume Confirmation (average volume) ===
    volume_12h = df_12h['volume'].values
    vol_ma_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    
    # === 12h ATR for trailing stop (14-period) ===
    tr1_12h = high_12h - low_12h
    tr2_12h = np.abs(high_12h - np.roll(close_12h, 1))
    tr3_12h = np.abs(low_12h - np.roll(close_12h, 1))
    tr2_12h[0] = tr1_12h[0]
    tr3_12h[0] = tr1_12h[0]
    tr_12h = np.maximum(tr1_12h, np.maximum(tr2_12h, tr3_12h))
    atr_12h = pd.Series(tr_12h).rolling(window=14, min_periods=14).mean().values
    atr_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_12h)
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 50
    
    # Track position and entry price for trailing stop
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(bb_width_percentile_aligned[i]) or 
            np.isnan(rsi_aligned[i]) or
            np.isnan(vol_ma_12h_aligned[i]) or
            np.isnan(atr_12h_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        bb_width_pct = bb_width_percentile_aligned[i]
        rsi_val = rsi_aligned[i]
        vol_ma_val = vol_ma_12h_aligned[i]
        atr_val = atr_12h_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 12h average volume
        vol_confirm = volume[i] > vol_ma_val * 1.5
        
        # Squeeze condition: BB width < 20th percentile (low volatility)
        squeeze_condition = bb_width_pct < 20
        
        # === TRAILING STOP LOGIC ===
        if position == 1:  # Long position
            # Update highest price since entry
            if price > highest_since_entry:
                highest_since_entry = price
            # Trail stop: exit if price drops 1.5*ATR from highest
            if atr_val > 0 and price < highest_since_entry - 1.5 * atr_val:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
                continue
        
        elif position == -1:  # Short position
            # Update lowest price since entry
            if price < lowest_since_entry or lowest_since_entry == 0:
                lowest_since_entry = price
            # Trail stop: exit if price rises 1.5*ATR from lowest
            if atr_val > 0 and price > lowest_since_entry + 1.5 * atr_val:
                signals[i] = 0.0
                position = 0
                lowest_since_entry = 0.0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Long when: squeeze AND RSI < 30 (oversold) AND volume confirmation
            if squeeze_condition and rsi_val < 30 and vol_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = price
                highest_since_entry = price
                continue
            # Short when: squeeze AND RSI > 70 (overbought) AND volume confirmation
            elif squeeze_condition and rsi_val > 70 and vol_confirm:
                signals[i] = -0.25
                position = -1
                entry_price = price
                lowest_since_entry = price
                continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "12h_BBWidthPercentile20_RSI30_70_Volume1.5x_ATRTrail_1.5x"
timeframe = "12h"
leverage = 1.0