#!/usr/bin/env python3
"""
6h_Adaptive_Kelly_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike
Hypothesis: Combines proven Camarilla R3/S3 breakout with 1d trend filter and volume confirmation, using adaptive Kelly position sizing based on recent win rate and volatility to reduce drawdown in bear markets while maximizing returns in bull markets. Targets 12-37 trades/year on 6h timeframe with discrete position sizing to minimize fee drag. Works in both bull and bear markets by following 1d trend direction and scaling exposure based on strategy performance.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for HTF filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Previous day's OHLC for Camarilla levels (R3/S3 = breakout/continuation levels)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_vals = df_1d['close'].values
    
    # Calculate Camarilla levels: R3, S3 (breakout/continuation levels)
    rng = high_1d - low_1d
    camarilla_r3 = close_1d_vals + (rng * 1.1 / 4)   # R3 level
    camarilla_s3 = close_1d_vals - (rng * 1.1 / 4)   # S3 level
    
    # Align Camarilla levels to 6h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume confirmation: volume > 2.0x 20-period average (dynamic threshold)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25  # Base position size (25% of capital)
    max_size = 0.35   # Maximum position size
    
    # Track recent trades for adaptive Kelly (last 20 trades)
    lookback_trades = 20
    trade_returns = []  # Store returns of recent closed trades
    entry_price = 0.0
    entry_time = 0
    
    # Warmup: max of calculations (20 for vol, 34 for 1d EMA)
    start_idx = max(20, 34)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        close_val = close[i]
        camarilla_r3_val = camarilla_r3_aligned[i]
        camarilla_s3_val = camarilla_s3_aligned[i]
        ema_34_val = ema_34_1d_aligned[i]
        vol_spike = volume_spike[i]
        
        # Determine 1d trend: bullish if price > EMA34, bearish if price < EMA34
        bullish_1d = close_val > ema_34_val
        bearish_1d = close_val < ema_34_val
        
        # Entry conditions: breakout of Camarilla R3/S3 in trend direction with volume spike
        long_entry = (close_val > camarilla_r3_val) and bullish_1d and vol_spike
        short_entry = (close_val < camarilla_s3_val) and bearish_1d and vol_spike
        
        # Exit conditions: mean reversion to midpoint or trend change
        mid_point = (camarilla_r3_val + camarilla_s3_val) / 2
        
        if position == 0:
            # Flat - look for entry
            if long_entry:
                # Calculate adaptive Kelly size based on recent performance
                kelly_fraction = calculate_kelly_fraction(trade_returns, lookback_trades)
                size = min(base_size * (1 + kelly_fraction), max_size)
                signals[i] = size
                position = 1
                entry_price = close_val
                entry_time = i
            elif short_entry:
                # Calculate adaptive Kelly size based on recent performance
                kelly_fraction = calculate_kelly_fraction(trade_returns, lookback_trades)
                size = min(base_size * (1 + kelly_fraction), max_size)
                signals[i] = -size
                position = -1
                entry_price = close_val
                entry_time = i
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit on mean reversion to midpoint or trend change
            if close_val < mid_point or not bullish_1d:
                # Calculate trade return for Kelly update
                if entry_price > 0:
                    trade_return = (close_val - entry_price) / entry_price
                    trade_returns.append(trade_return)
                    # Keep only recent trades
                    if len(trade_returns) > lookback_trades:
                        trade_returns = trade_returns[-lookback_trades:]
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = base_size
        elif position == -1:
            # Short - exit on mean reversion to midpoint or trend change
            if close_val > mid_point or not bearish_1d:
                # Calculate trade return for Kelly update
                if entry_price > 0:
                    trade_return = (entry_price - close_val) / entry_price
                    trade_returns.append(trade_return)
                    # Keep only recent trades
                    if len(trade_returns) > lookback_trades:
                        trade_returns = trade_returns[-lookback_trades:]
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -base_size
    
    return signals

def calculate_kelly_fraction(returns, lookback):
    """Calculate Kelly fraction based on recent trade returns"""
    if len(returns) < 5:  # Need minimum samples
        return 0.0
    
    returns_array = np.array(returns)
    win_rate = np.mean(returns_array > 0)
    if win_rate == 0 or win_rate == 1:
        return 0.0
    
    avg_win = np.mean(returns_array[returns_array > 0]) if np.any(returns_array > 0) else 0
    avg_loss = np.mean(np.abs(returns_array[returns_array < 0])) if np.any(returns_array < 0) else 0
    
    if avg_loss == 0:
        return 0.0
    
    # Kelly formula: f = (bp - q) / b where b = avg_win/avg_loss, p = win_rate, q = loss_rate
    b = avg_win / avg_loss
    p = win_rate
    q = 1 - win_rate
    kelly = (b * p - q) / b
    
    # Limit Kelly fraction to reasonable range [-0.5, 0.5] and scale down for safety
    kelly = np.clip(kelly, -0.5, 0.5)
    return kelly * 0.5  # Use half-Kelly for reduced volatility

name = "6h_Adaptive_Kelly_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0