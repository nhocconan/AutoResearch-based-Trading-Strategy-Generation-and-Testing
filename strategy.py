#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_camarilla_breakout_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return signals
    
    # Calculate weekly Camarilla pivots
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Pivot point and levels
    pivot_1w = (high_1w + low_1w + close_1w) / 3
    range_1w = high_1w - low_1w
    
    # Camarilla levels (standard multipliers)
    r4_1w = close_1w + range_1w * 1.1 / 2
    s4_1w = close_1w - range_1w * 1.1 / 2
    
    # Align weekly pivots to daily timeframe
    r4_1w_aligned = align_htf_to_ltf(prices, df_1w, r4_1w)
    s4_1w_aligned = align_htf_to_ltf(prices, df_1w, s4_1w)
    
    # Volume confirmation: volume > 1.5x 10-period average
    vol_ma_10 = pd.Series(volume).rolling(window=10, min_periods=10).mean().values
    
    # ATR for stop loss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    for i in range(10, n):
        # Skip if any required data is invalid
        if (np.isnan(r4_1w_aligned[i]) or np.isnan(s4_1w_aligned[i]) or
            np.isnan(vol_ma_10[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        volume_current = volume[i]
        r4 = r4_1w_aligned[i]
        s4 = s4_1w_aligned[i]
        atr_val = atr[i]
        
        # Volume confirmation
        volume_confirmed = volume_current > 1.5 * vol_ma_10[i]
        
        # Entry signals
        long_signal = False
        short_signal = False
        
        # Long: price breaks above R4 with volume
        if price_high > r4 and volume_confirmed:
            long_signal = True
        
        # Short: price breaks below S4 with volume
        if price_low < s4 and volume_confirmed:
            short_signal = True
        
        # Exit conditions
        # Exit on opposite level touch (mean reversion)
        exit_long = position == 1 and price_low <= s4
        exit_short = position == -1 and price_high >= r4
        
        # Stop loss conditions
        stop_long = position == 1 and price_low < (entry_price - 2.0 * atr_val)
        stop_short = position == -1 and price_high > (entry_price + 2.0 * atr_val)
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            entry_price = price_close
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            entry_price = price_close
            signals[i] = -0.25
        elif position == 1 and (exit_long or stop_long):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (exit_short or stop_short):
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

# Hypothesis: 1d Camarilla breakout strategy with weekly context, volume confirmation, and ATR stop loss.
# Uses weekly Camarilla levels (R4/S4) as key support/resistance levels for breakout trading.
# Enters long when price breaks above weekly R4 with volume confirmation (>1.5x 10-day average volume).
# Enters short when price breaks below weekly S4 with volume confirmation.
# Exits when price returns to the opposite level (S4 for longs, R4 for shorts) or ATR stop loss (2x) is hit.
# Weekly timeframe provides broader market context to avoid false breakouts in daily noise.
# Designed to work in both bull and bear markets by capturing strong directional moves that break weekly structures.
# Target: 10-20 trades per year to minimize fee decay while capturing significant breakouts.
# Works in both bull and bear markets by trading breakouts in either direction.