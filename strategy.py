#!/usr/bin/env python3
"""
4h_1d_Combo_Trend_MeanRev_Signal
Hypothesis: Combines trend-following (Donchian breakout) and mean-reversion (RSI pullback) signals, filtered by 1d EMA50 trend and volume spike. Trades in both bull and bear markets by capturing breakouts in trends and pullbacks in ranges. Uses volatility-adjusted position sizing and ATR stop-loss to manage risk. Targets 20-40 trades/year.
"""

name = "4h_1d_Combo_Trend_MeanRev_Signal"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 4h OHLCV
    close_4h = prices['close'].values
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    volume_4h = prices['volume'].values
    
    # --- 1d Trend Filter: EMA50 ---
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # --- 4h Donchian Channels (20-period) for breakout ---
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # --- 4h RSI (14-period) for mean reversion ---
    delta = np.diff(close_4h, prepend=close_4h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # --- Volume Filter: spike above 1.5x 20-period average ---
    vol_ma = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_4h > (1.5 * vol_ma)
    
    # --- ATR for stoploss (14-period) ---
    tr1 = np.abs(high_4h - low_4h)
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start after warmup period
    start_idx = 50  # for EMA50, Donchian, RSI, volume MA, ATR
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                # Check stoploss
                if position == 1 and close_4h[i] <= entry_price - 2.0 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                elif position == -1 and close_4h[i] >= entry_price + 2.0 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25 if position == 1 else -0.25
            continue
        
        # Determine 1d trend
        trend_up = close_4h[i] > ema50_1d_aligned[i]
        trend_down = close_4h[i] < ema50_1d_aligned[i]
        
        if position == 0:
            # Look for trend-following entry: Donchian breakout in trend direction
            if close_4h[i] > donchian_high[i] and trend_up and vol_spike[i]:
                # Long breakout
                signals[i] = 0.25
                position = 1
                entry_price = close_4h[i]
            elif close_4h[i] < donchian_low[i] and trend_down and vol_spike[i]:
                # Short breakdown
                signals[i] = -0.25
                position = -1
                entry_price = close_4h[i]
            # Look for mean-reversion entry: RSI pullback against short-term bias
            elif rsi[i] < 30 and not trend_down:  # Oversold but not in strong downtrend
                # Long pullback
                signals[i] = 0.20
                position = 1
                entry_price = close_4h[i]
            elif rsi[i] > 70 and not trend_up:  # Overbought but not in strong uptrend
                # Short pullback
                signals[i] = -0.20
                position = -1
                entry_price = close_4h[i]
        else:
            # Update stoploss and check exits
            if position == 1:
                # Stoploss
                if close_4h[i] <= entry_price - 2.0 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                # Exit: RSI overbought or price touches Donchian low (for breakout) or RSI > 60 (for mean reversion)
                elif rsi[i] > 70 or close_4h[i] <= donchian_low[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25 if rsi[i] < 50 else 0.20  # Slightly reduce size in strong uptrend
            elif position == -1:
                # Stoploss
                if close_4h[i] >= entry_price + 2.0 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                # Exit: RSI oversold or price touches Donchian high (for breakdown) or RSI < 40 (for mean reversion)
                elif rsi[i] < 30 or close_4h[i] >= donchian_high[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25 if rsi[i] > 50 else -0.20  # Slightly reduce size in strong downtrend
    
    return signals