#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Squeeze + Volume Breakout with 1d EMA34 trend filter
# Bollinger Squeeze identifies low volatility periods preceding breakouts
# Squeeze occurs when Bollinger Band width < Keltner Channel width
# Breakout direction confirmed by close outside Bollinger Bands + volume surge
# 1d EMA34 provides higher timeframe trend bias to avoid counter-trend trades
# Works in both bull and bear markets by capturing volatility expansion moves
# Target: 50-150 total trades over 4 years (12-37/year) with disciplined entries
name = "6h_BollingerSqueeze_1dEMA34_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Bollinger Bands (20, 2)
    bb_period = 20
    bb_std = 2
    sma = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    bb_std_dev = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    bb_upper = sma + (bb_std_dev * bb_std)
    bb_lower = sma - (bb_std_dev * bb_std)
    bb_width = bb_upper - bb_lower
    
    # Keltner Channel (20, 1.5 ATR)
    kc_period = 20
    kc_multiplier = 1.5
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(np.maximum(tr1, tr2), tr3)])
    atr = pd.Series(tr).rolling(window=kc_period, min_periods=kc_period).mean().values
    kc_middle = pd.Series(close).rolling(window=kc_period, min_periods=kc_period).mean().values
    kc_upper = kc_middle + (atr * kc_multiplier)
    kc_lower = kc_middle - (atr * kc_multiplier)
    kc_width = kc_upper - kc_lower
    
    # Squeeze condition: BB width < KC width
    squeeze = bb_width < kc_width
    
    # Breakout conditions
    breakout_up = close > bb_upper
    breakout_down = close < bb_lower
    
    # Volume confirmation: volume > 1.5 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(sma[i]) or np.isnan(bb_width[i]) or np.isnan(kc_width[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Look for breakout after squeeze with volume confirmation and trend alignment
            if (squeeze[i-1] and  # Was in squeeze on previous bar
                volume_confirm[i] and  # Current bar has volume confirmation
                breakout_up[i] and  # Price broke above upper BB
                close[i] > ema_34_1d_aligned[i]):  # Above 1d EMA34 (uptrend bias)
                signals[i] = 0.25
                position = 1
            elif (squeeze[i-1] and  # Was in squeeze on previous bar
                  volume_confirm[i] and  # Current bar has volume confirmation
                  breakout_down[i] and  # Price broke below lower BB
                  close[i] < ema_34_1d_aligned[i]):  # Below 1d EMA34 (downtrend bias)
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price returns to middle Bollinger Band or trend breaks
            if (close[i] <= sma[i]) or (close[i] < ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price returns to middle Bollinger Band or trend breaks
            if (close[i] >= sma[i]) or (close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals