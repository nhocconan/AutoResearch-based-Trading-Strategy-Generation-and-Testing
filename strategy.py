#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Bollinger Bands squeeze breakout with weekly trend filter and volume confirmation
# Bollinger Band width < 20th percentile indicates low volatility (squeeze).
# Breakout occurs when price closes outside bands with volume > 1.5x average.
# Weekly EMA50 trend filter: only take long breaks in weekly uptrend (price > EMA50),
# short breaks in weekly downtrend (price < EMA50).
# Works in ranging markets (squeeze breakouts) and trending markets (continuation).
# Target: 50-150 total trades over 4 years (12-37/year) with disciplined entries.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 12h Bollinger Bands (20, 2) ===
    bb_period = 20
    bb_std = 2
    sma = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    std_dev = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper_band = sma + (bb_std * std_dev)
    lower_band = sma - (bb_std * std_dev)
    bb_width = ((upper_band - lower_band) / sma) * 100  # percentage
    
    # === 12h Bollinger Band width percentile (200 lookback) ===
    bb_width_percentile = pd.Series(bb_width).rolling(window=200, min_periods=20).rank(pct=True).values * 100
    
    # === 12h volume ratio for confirmation ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma_20
    
    # === Weekly trend filter (EMA50) ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 200
    
    # Track position and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(bb_width_percentile[i]) or np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(vol_ratio[i]) or np.isnan(upper_band[i]) or np.isnan(lower_band[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        bb_width_pct = bb_width_percentile[i]
        weekly_ema = ema_50_1w_aligned[i]
        vol = vol_ratio[i]
        upper = upper_band[i]
        lower = lower_band[i]
        
        # === STOPLOSS LOGIC (close-based) ===
        if position == 1:  # Long position
            if price < entry_price - 2.0 * (upper - lower):  # 2x BB width as SL
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        elif position == -1:  # Short position
            if price > entry_price + 2.0 * (upper - lower):  # 2x BB width as SL
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when price touches opposite band or squeeze ends
            if price >= lower or bb_width_pct > 50:  # touch lower band or volatility expansion
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        elif position == -1:  # Short position
            # Exit when price touches opposite band or squeeze ends
            if price <= upper or bb_width_pct > 50:  # touch upper band or volatility expansion
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Squeeze condition: BB width in lowest 20%
            is_squeeze = bb_width_pct < 20
            # Volume confirmation
            vol_confirm = vol > 1.5
            # Weekly trend alignment
            weekly_uptrend = price > weekly_ema
            weekly_downtrend = price < weekly_ema
            
            # Long: squeeze breakout up in weekly uptrend
            if is_squeeze and vol_confirm and weekly_uptrend and price > upper:
                signals[i] = 0.25
                position = 1
                entry_price = price
                continue
            # Short: squeeze breakout down in weekly downtrend
            elif is_squeeze and vol_confirm and weekly_downtrend and price < lower:
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

name = "12h_BollingerSqueeze_WeeklyTrend_VolumeBreakout_v1"
timeframe = "12h"
leverage = 1.0