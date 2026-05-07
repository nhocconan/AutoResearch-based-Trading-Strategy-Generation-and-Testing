#!/usr/bin/env python3
name = "1d_Adaptive_Kelly_Trend_With_Volume_Confirmation"
timeframe = "1d"
leverage = 1.0

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
    
    # Load weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Daily indicators
    # EMA 50 for trend direction
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    # ATR for volatility
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    # Volume average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Weekly indicators for trend filter
    # Weekly EMA 20
    ema_20_1w = pd.Series(df_1w['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    # Weekly RSI 14 for momentum
    delta = np.diff(df_1w['close'])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_14_1w = (100 - (100 / (1 + rs))).values
    # Align weekly indicators to daily
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    rsi_14_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_14_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50[i]) or np.isnan(atr[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(ema_20_1w_aligned[i]) or np.isnan(rsi_14_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend condition: price relative to EMA50
        price_above_ema50 = close[i] > ema_50[i]
        price_below_ema50 = close[i] < ema_50[i]
        
        # Weekly trend and momentum
        weekly_uptrend = ema_20_1w_aligned[i] > ema_20_1w_aligned[i-1]
        weekly_downtrend = ema_20_1w_aligned[i] < ema_20_1w_aligned[i-1]
        weekly_momentum_up = rsi_14_1w_aligned[i] > 50
        weekly_momentum_down = rsi_14_1w_aligned[i] < 50
        
        # Volume confirmation
        vol_spike = volume[i] > vol_ma[i] * 1.5
        
        # Kelly-inspired position sizing based on trend strength
        # Calculate trend strength as distance from EMA50 normalized by ATR
        if atr[i] > 0:
            trend_strength = abs(close[i] - ema_50[i]) / atr[i]
            # Scale position size: stronger trend = larger position (capped)
            base_size = 0.25
            size_multiplier = min(1.0 + (trend_strength - 1.0) * 0.2, 1.4)  # Max 40% increase
            position_size = base_size * size_multiplier
            position_size = min(position_size, 0.35)  # Hard cap at 0.35
        else:
            position_size = 0.25
        
        if position == 0:
            # Long: price above EMA50, weekly uptrend, momentum up, volume spike
            if price_above_ema50 and weekly_uptrend and weekly_momentum_up and vol_spike:
                signals[i] = position_size
                position = 1
            # Short: price below EMA50, weekly downtrend, momentum down, volume spike
            elif price_below_ema50 and weekly_downtrend and weekly_momentum_down and vol_spike:
                signals[i] = -position_size
                position = -1
        elif position == 1:
            # Exit: price crosses below EMA50 or weekly trend/momentum turns down
            if not price_above_ema50 or not weekly_uptrend or not weekly_momentum_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit: price crosses above EMA50 or weekly trend/momentum turns up
            if not price_below_ema50 or not weekly_downtrend or not weekly_momentum_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -position_size
    
    return signals

# Hypothesis: Adaptive position sizing based on trend strength with weekly trend filter
# - Uses daily EMA50 as primary trend filter
# - Weekly EMA20 and RSI14 for higher timeframe trend and momentum confirmation
# - Volume spike (1.5x average) required for entry to avoid false signals
# - Position size adapts to trend strength: stronger trends get larger positions (up to 0.35)
# - Exits when price crosses EMA50 or weekly trend/momentum deteriorates
# - Designed to work in both bull and bear markets by following the trend
# - Adaptive sizing reduces risk in weak trends while capturing strong moves
# - Volume confirmation filters out low-quality breakouts
# - Weekly alignment ensures we only trade in the direction of higher timeframe momentum
# - Target: 20-50 trades per year to minimize fee drag while capturing significant moves