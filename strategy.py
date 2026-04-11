#!/usr/bin/env python3
# 12h_1w_camarilla_pivot_volume_v1
# Strategy: 12h Camarilla pivot reversal with 1w trend filter and volume confirmation
# Timeframe: 12h
# Leverage: 1.0
# Hypothesis: Camarilla levels act as strong support/resistance. In trending markets (1w EMA), price rejects at levels (fade). In ranging markets, price breaks levels (breakout). Volume confirms institutional participation. Designed for low trade frequency (~15-30/year) to minimize fee drift.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1w_camarilla_pivot_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1w EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # 12h volume average (20-period) for confirmation
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_avg_20[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Need at least 2 days of data for Camarilla calculation
        if i < 48:  # 2 days * 24h / 12h = 4 periods, but use buffer
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Calculate Camarilla levels from previous day (24h ago = 2 periods back)
        # Use daily high/low/close from 24h ago
        prev_day_idx = i - 2
        if prev_day_idx < 0:
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
            
        # Get daily OHLC from 12h data: need to resample conceptually
        # Since we're on 12h timeframe, 2 periods back = previous day's close
        # For simplicity, use the 12h bar from 2 periods ago as proxy for daily close
        # In practice, we'd use actual daily data, but we approximate with available data
        # Better approach: use 1d data for Camarilla, but we're constrained to 1w HTF
        # Instead, calculate pivots from previous 12h bar's range (approximation)
        lookback_idx = i - 1
        if lookback_idx < 0:
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
            
        # Use previous bar's high/low/close for Camarilla (simplified)
        ph = high[lookback_idx]
        pl = low[lookback_idx]
        pc = close[lookback_idx]
        
        # Camarilla levels
        range_val = ph - pl
        if range_val <= 0:
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
            
        # Key levels: L3, L3, H3, H4
        l3 = pc - (range_val * 1.1 / 4)
        h3 = pc + (range_val * 1.1 / 4)
        l4 = pc - (range_val * 1.1 / 2)
        h4 = pc + (range_val * 1.1 / 2)
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume[i] > 1.5 * vol_avg_20[i]
        
        # Trend filter: 1w EMA50
        trend_bullish = close[i] > ema_50_1w_aligned[i]
        trend_bearish = close[i] < ema_50_1w_aligned[i]
        
        # Entry logic: fade at H3/L3 in trend, break at H4/L4 in range
        # Determine market regime: use price position relative to EMA
        # If price far from EMA, likely trending; if near, likely ranging
        ema_dist = abs(close[i] - ema_50_1w_aligned[i]) / ema_50_1w_aligned[i]
        is_trending = ema_dist > 0.02  # 2% deviation from EMA
        
        # Long conditions
        long_signal = False
        if is_trending:
            # In trend: fade at L3 (support)
            if close[i] <= l3 and close[i] > l4 and vol_confirm and trend_bullish:
                long_signal = True
        else:
            # In range: break above H4 (breakout)
            if close[i] > h4 and vol_confirm:
                long_signal = True
                
        # Short conditions
        short_signal = False
        if is_trending:
            # In trend: fade at H3 (resistance)
            if close[i] >= h3 and close[i] < h4 and vol_confirm and trend_bearish:
                short_signal = True
        else:
            # In range: break below L4 (breakdown)
            if close[i] < l4 and vol_confirm:
                short_signal = True
        
        # Exit conditions: opposite signal or volatility expansion
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long: short signal or stop loss (simplified: opposite Camarilla)
            if short_signal or close[i] >= h3:
                exit_long = True
        elif position == -1:
            # Exit short: long signal or stop loss
            if long_signal or close[i] <= l3:
                exit_short = True
        
        # Update signals
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long:
            position = 0
            signals[i] = 0.0
        elif exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals