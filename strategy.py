#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 12h data (HTF for trend and pivot levels) ===
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # 12h EMA34 for trend filter
    close_12h_series = pd.Series(close_12h)
    ema_34_12h = close_12h_series.ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # === 6h data (primary timeframe) ===
    # 6h Donchian(20) for entry/exit levels
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_upper = high_20
    donchian_lower = low_20
    
    # === 1d data (HTF for regime and pivot levels) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 1d ATR for volatility regime
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = np.inf
    tr3[0] = np.inf
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # === 12h Pivot Points (using 12h high/low/close) ===
    pivot_12h = (high_12h + low_12h + close_12h) / 3.0
    r1_12h = 2 * pivot_12h - low_12h
    s1_12h = 2 * pivot_12h - high_12h
    r2_12h = pivot_12h + (high_12h - low_12h)
    s2_12h = pivot_12h - (high_12h - low_12h)
    r3_12h = pivot_12h + 2 * (high_12h - low_12h)
    s3_12h = pivot_12h - 2 * (high_12h - low_12h)
    r4_12h = pivot_12h + 3 * (high_12h - low_12h)
    s4_12h = pivot_12h - 3 * (high_12h - low_12h)
    
    # Align pivot levels
    pivot_12h_aligned = align_htf_to_ltf(prices, df_12h, pivot_12h)
    r1_12h_aligned = align_htf_to_ltf(prices, df_12h, r1_12h)
    s1_12h_aligned = align_htf_to_ltf(prices, df_12h, s1_12h)
    r4_12h_aligned = align_htf_to_ltf(prices, df_12h, r4_12h)
    s4_12h_aligned = align_htf_to_ltf(prices, df_12h, s4_12h)
    
    # === 1h data for RSI and volume (entry timing) ===
    df_1h = get_htf_data(prices, '1h')
    close_1h = df_1h['close'].values
    volume_1h = df_1h['volume'].values
    
    # RSI(14) on 1h
    delta = np.diff(close_1h, prepend=close_1h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1h = 100 - (100 / (1 + rs))
    rsi_1h_aligned = align_htf_to_ltf(prices, df_1h, rsi_1h)
    
    # Volume spike detection on 1h
    vol_ma_10_1h = pd.Series(volume_1h).rolling(window=10, min_periods=10).mean().values
    vol_ratio_1h = volume_1h / vol_ma_10_1h
    vol_ratio_1h_aligned = align_htf_to_ltf(prices, df_1h, vol_ratio_1h)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators have valid data
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_12h_aligned[i]) or np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(atr_1d_aligned[i]) or np.isnan(rsi_1h_aligned[i]) or np.isnan(vol_ratio_1h_aligned[i]) or
            np.isnan(pivot_12h_aligned[i]) or np.isnan(r1_12h_aligned[i]) or np.isnan(s1_12h_aligned[i]) or
            np.isnan(r4_12h_aligned[i]) or np.isnan(s4_12h_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        price = close[i]
        ema_34_12h_val = ema_34_12h_aligned[i]
        atr_1d_val = atr_1d_aligned[i]
        rsi_1h_val = rsi_1h_aligned[i]
        vol_ratio_1h_val = vol_ratio_1h_aligned[i]
        pivot_val = pivot_12h_aligned[i]
        r1_val = r1_12h_aligned[i]
        s1_val = s1_12h_aligned[i]
        r4_val = r4_12h_aligned[i]
        s4_val = s4_12h_aligned[i]
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when price closes below S1 or RSI overbought
            if (price < s1_val) or (rsi_1h_val > 70):
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit when price closes above R1 or RSI oversold
            if (price > r1_val) or (rsi_1h_val < 30):
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Only trade during session
            if in_session:
                # LONG: Price breaks above R1 AND above 12h EMA34 (trend filter)
                # AND RSI not overbought AND volume spike AND volatility not too high
                if (price > r1_val) and (price > ema_34_12h_val) and (rsi_1h_val < 60) and \
                   (vol_ratio_1h_val > 2.0) and (atr_1d_val < np.percentile(atr_1d_aligned[:i+1], 80)):
                    signals[i] = 0.25
                    position = 1
                    continue
                
                # SHORT: Price breaks below S1 AND below 12h EMA34 (trend filter)
                # AND RSI not oversold AND volume spike AND volatility not too high
                elif (price < s1_val) and (price < ema_34_12h_val) and (rsi_1h_val > 40) and \
                     (vol_ratio_1h_val > 2.0) and (atr_1d_val < np.percentile(atr_1d_aligned[:i+1], 80)):
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

name = "6h_Pivot_R1_S1_Breakout_EMA34_Volume"
timeframe = "6h"
leverage = 1.0