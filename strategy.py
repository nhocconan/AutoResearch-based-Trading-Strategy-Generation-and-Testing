#!/usr/bin/env python3
"""
12h_1d_Adaptive_Kelly_Camarilla_Breakout
Hypothesis: Use Kelly criterion sizing based on recent Kelly fraction from winning trades.
Enter on 1d Camarilla R1/S1 breakouts with volume confirmation and trend filter from 4h EMA34.
Exit at opposite Camarilla level (S1 for long, R1 for short).
Kelly sizing reduces risk after losses and increases after wins, improving risk-adjusted returns.
Designed for 12h timeframe to target 50-150 total trades over 4 years with controlled risk.
Works in bull markets by buying breakouts and in bear markets by selling breakdowns.
Volume and trend filters reduce false breakouts and whipsaws.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data once for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla pivot levels (based on previous day)
    # PP = (H + L + C) / 3
    # R1 = C + (H - L) * 1.1 / 12
    # S1 = C - (H - L) * 1.1 / 12
    pp = np.full_like(close_1d, np.nan)
    r1 = np.full_like(close_1d, np.nan)
    s1 = np.full_like(close_1d, np.nan)
    
    for i in range(1, len(high_1d)):
        pp[i] = (high_1d[i-1] + low_1d[i-1] + close_1d[i-1]) / 3.0
        r1[i] = close_1d[i-1] + (high_1d[i-1] - low_1d[i-1]) * 1.1 / 12.0
        s1[i] = close_1d[i-1] - (high_1d[i-1] - low_1d[i-1]) * 1.1 / 12.0
    
    # Shift to align with current day (levels are based on previous day)
    pp = np.roll(pp, 1)
    r1 = np.roll(r1, 1)
    s1 = np.roll(s1, 1)
    pp[0] = np.nan
    r1[0] = np.nan
    s1[0] = np.nan
    
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # 4h EMA34 for trend filter (more responsive than 50)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 34:
        return np.zeros(n)
    close_4h = df_4h['close'].values
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Track trade performance for Kelly sizing
    trade_returns = []  # Store returns of closed trades
    kelly_fraction = 0.25  # Start with conservative 25% Kelly
    max_kelly = 0.35       # Cap Kelly fraction at 35%
    min_kelly = 0.10       # Minimum Kelly fraction
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(ema_34_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                # Calculate trade return when closing
                if entry_price > 0:
                    if position == 1:  # Was long
                        trade_ret = (prices['close'].iloc[i] - entry_price) / entry_price
                    else:  # Was short
                        trade_ret = (entry_price - prices['close'].iloc[i]) / entry_price
                    trade_returns.append(trade_ret)
                    # Update Kelly fraction based on recent trades
                    if len(trade_returns) >= 5:
                        recent = trade_returns[-5:]
                        wins = [r for r in recent if r > 0]
                        losses = [r for r in recent if r < 0]
                        win_rate = len(wins) / len(recent) if len(recent) > 0 else 0.5
                        avg_win = np.mean(wins) if wins else 0.0
                        avg_loss = np.mean([abs(l) for l in losses]) if losses else 0.0
                        if avg_loss > 0 and win_rate > 0:
                            kelly = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
                            kelly = max(min_kelly, min(max_kelly, kelly))  # Clamp
                            kelly_fraction = kelly * 0.5  # Use half-Kelly for safety
                    entry_price = 0.0
            continue
        
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume filter: current volume > 1.3 * 20-period average
        if i >= 20:
            vol_ma = prices['volume'].iloc[i-20:i].mean()
            volume_ok = volume > 1.3 * vol_ma
        else:
            volume_ok = False
        
        if position == 0:
            # Long conditions: break above R1 + volume confirmation + price above EMA34
            if price > r1_aligned[i] and volume_ok and price > ema_34_aligned[i]:
                signals[i] = kelly_fraction
                position = 1
                entry_price = price
            # Short conditions: break below S1 + volume confirmation + price below EMA34
            elif price < s1_aligned[i] and volume_ok and price < ema_34_aligned[i]:
                signals[i] = -kelly_fraction
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: price crosses back below S1 (opposite level)
            if price < s1_aligned[i]:
                signals[i] = 0.0
                position = 0
                # Calculate trade return when closing
                if entry_price > 0:
                    trade_ret = (price - entry_price) / entry_price
                    trade_returns.append(trade_ret)
                    # Update Kelly fraction based on recent trades
                    if len(trade_returns) >= 5:
                        recent = trade_returns[-5:]
                        wins = [r for r in recent if r > 0]
                        losses = [r for r in recent if r < 0]
                        win_rate = len(wins) / len(recent) if len(recent) > 0 else 0.5
                        avg_win = np.mean(wins) if wins else 0.0
                        avg_loss = np.mean([abs(l) for l in losses]) if losses else 0.0
                        if avg_loss > 0 and win_rate > 0:
                            kelly = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
                            kelly = max(min_kelly, min(max_kelly, kelly))  # Clamp
                            kelly_fraction = kelly * 0.5  # Use half-Kelly for safety
                    entry_price = 0.0
            else:
                signals[i] = kelly_fraction
        
        elif position == -1:
            # Short exit: price crosses back above R1 (opposite level)
            if price > r1_aligned[i]:
                signals[i] = 0.0
                position = 0
                # Calculate trade return when closing
                if entry_price > 0:
                    trade_ret = (entry_price - price) / entry_price
                    trade_returns.append(trade_ret)
                    # Update Kelly fraction based on recent trades
                    if len(trade_returns) >= 5:
                        recent = trade_returns[-5:]
                        wins = [r for r in recent if r > 0]
                        losses = [r for r in recent if r < 0]
                        win_rate = len(wins) / len(recent) if len(recent) > 0 else 0.5
                        avg_win = np.mean(wins) if wins else 0.0
                        avg_loss = np.mean([abs(l) for l in losses]) if losses else 0.0
                        if avg_loss > 0 and win_rate > 0:
                            kelly = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
                            kelly = max(min_kelly, min(max_kelly, kelly))  # Clamp
                            kelly_fraction = kelly * 0.5  # Use half-Kelly for safety
                    entry_price = 0.0
            else:
                signals[i] = -kelly_fraction
    
    return signals

name = "12h_1d_Adaptive_Kelly_Camarilla_Breakout"
timeframe = "12h"
leverage = 1.0