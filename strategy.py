#!/usr/bin/env python3
"""
6h_Adaptive_Kelly_Camarilla_R3S3_Breakout_1dTrend
Hypothesis: Camarilla R3/S3 breakout on 6h with 1d EMA34 trend filter, volume confirmation, and adaptive Kelly position sizing.
Uses Kelly fraction based on recent win rate and profit factor to scale position size (0.0-0.35).
Adaptive sizing reduces exposure during losing streaks and increases during winning periods.
Works in bull/bear via 1d trend filter and avoids choppy markets via volume confirmation.
Target: 12-37 trades/year on 6h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    close_1d_series = pd.Series(close_1d)
    ema_34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from prior 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla R3, S3, R4, S4 for each 1d bar
    rng = high_1d - low_1d
    r3 = close_1d + 1.125 * rng
    s3 = close_1d - 1.125 * rng
    r4 = close_1d + 1.5 * rng
    s4 = close_1d - 1.5 * rng
    
    # Align to 6h timeframe (wait for 1d bar to close)
    r3_6h = align_htf_to_ltf(prices, df_1d, r3)
    s3_6h = align_htf_to_ltf(prices, df_1d, s3)
    r4_6h = align_htf_to_ltf(prices, df_1d, r4)
    s4_6h = align_htf_to_ltf(prices, df_1d, s4)
    
    # Volume spike: volume > 1.5x 20-period median volume
    volume_series = pd.Series(volume)
    vol_median_20 = volume_series.rolling(window=20, min_periods=20).median().values
    volume_spike = volume > (1.5 * vol_median_20)
    
    # Track trades for adaptive Kelly sizing
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    trade_returns = []  # store realized returns for Kelly calculation
    
    # Warmup: need 34 for 1d EMA, 20 for volume median
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(r3_6h[i]) or
            np.isnan(s3_6h[i]) or
            np.isnan(r4_6h[i]) or
            np.isnan(s4_6h[i]) or
            np.isnan(vol_median_20[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema_34_val = ema_34_1d_aligned[i]
        vol_spike = volume_spike[i]
        
        # Calculate Kelly fraction from recent trade history
        kelly_fraction = 0.25  # default
        if len(trade_returns) >= 10:
            # Win rate
            wins = [r for r in trade_returns if r > 0]
            win_rate = len(wins) / len(trade_returns) if trade_returns else 0.5
            # Average win / average loss
            avg_win = np.mean(wins) if wins else 0
            losses = [abs(r) for r in trade_returns if r < 0]
            avg_loss = np.mean(losses) if losses else 0
            if avg_loss > 0 and win_rate > 0:
                win_loss_ratio = avg_win / avg_loss
                kelly = win_rate - ((1 - win_rate) / win_loss_ratio)
                kelly_fraction = max(0.05, min(0.35, kelly * 0.5))  # quarter Kelly, capped
        
        if position == 0:
            # Flat - look for entry
            # Long: price > R3 and volume spike, in uptrend (close > EMA34)
            long_entry = (close_val > r3_6h[i]) and vol_spike and (close_val > ema_34_val)
            # Short: price < S3 and volume spike, in downtrend (close < EMA34)
            short_entry = (close_val < s3_6h[i]) and vol_spike and (close_val < ema_34_val)
            
            if long_entry:
                signals[i] = kelly_fraction
                position = 1
                entry_price = close_val
            elif short_entry:
                signals[i] = -kelly_fraction
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit on trend reversal or at R4 (take profit)
            if close_val < ema_34_val or close_val > r4_6h[i]:
                # Calculate trade return
                if entry_price > 0:
                    trade_return = (close_val - entry_price) / entry_price
                    trade_returns.append(trade_return)
                    # Keep only last 50 trades for Kelly calculation
                    if len(trade_returns) > 50:
                        trade_returns = trade_returns[-50:]
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = kelly_fraction
        elif position == -1:
            # Short - exit on trend reversal or at S4 (take profit)
            if close_val > ema_34_val or close_val < s4_6h[i]:
                # Calculate trade return
                if entry_price > 0:
                    trade_return = (entry_price - close_val) / entry_price
                    trade_returns.append(trade_return)
                    # Keep only last 50 trades for Kelly calculation
                    if len(trade_returns) > 50:
                        trade_returns = trade_returns[-50:]
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -kelly_fraction
    
    return signals

name = "6h_Adaptive_Kelly_Camarilla_R3S3_Breakout_1dTrend"
timeframe = "6h"
leverage = 1.0