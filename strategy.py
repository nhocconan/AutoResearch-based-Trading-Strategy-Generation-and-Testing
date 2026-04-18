#!/usr/bin/env python3
"""
1d_Keltner_Squeeze_Breakout_v1
Hypothesis: In low-volatility regimes (BB/KC squeeze), price often breaks out with momentum.
We use daily timeframe with weekly trend filter to capture multi-day moves.
Long when: price breaks above upper Keltner + weekly trend up + volume spike.
Short when: price breaks below lower Keltner + weekly trend down + volume spike.
Exit when: opposite Keltner break or trend change.
Designed for 1d timeframe: ~10-25 trades/year per symbol (40-100 total over 4 years).
Works in bull/bear via weekly trend filter and volatility squeeze logic.
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
    
    # Get weekly data for trend filter (using 1w as HTF)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA20 and EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align weekly data to daily timeframe
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Daily Keltner Channel (20, 2.0)
    # Typical Price = (H+L+C)/3
    typical_price = (high + low + close) / 3.0
    atr_period = 20
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    atr = pd.Series(tr).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    
    ema_tp = pd.Series(typical_price).ewm(span=20, adjust=False, min_periods=20).mean().values
    kc_upper = ema_tp + 2.0 * atr
    kc_lower = ema_tp - 2.0 * atr
    
    # Bollinger Bands (20, 2.0) for squeeze detection
    sma_close = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_close = pd.Series(close).rolling(window=20, min_periods=20).std().values
    bb_upper = sma_close + 2.0 * std_close
    bb_lower = sma_close - 2.0 * std_close
    
    # Squeeze condition: BB inside KC (low volatility)
    squeeze = (bb_upper <= kc_upper) & (bb_lower >= kc_lower)
    
    # Volume confirmation (20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # need enough for weekly EMA50 and daily indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_20_1w_aligned[i]) or np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(kc_upper[i]) or np.isnan(kc_lower[i]) or
            np.isnan(squeeze[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Weekly trend conditions
        weekly_uptrend = ema_20_1w_aligned[i] > ema_50_1w_aligned[i]
        weekly_downtrend = ema_20_1w_aligned[i] < ema_50_1w_aligned[i]
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma_20[i]
        
        # Breakout conditions
        buy_breakout = close[i] > kc_upper[i] and squeeze[i]
        sell_breakout = close[i] < kc_lower[i] and squeeze[i]
        
        if position == 0:
            # Long: weekly uptrend + volume + buy breakout in squeeze
            if weekly_uptrend and vol_confirm and buy_breakout:
                signals[i] = 0.25
                position = 1
            # Short: weekly downtrend + volume + sell breakout in squeeze
            elif weekly_downtrend and vol_confirm and sell_breakout:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: weekly trend change, sell breakout, or squeeze release
            if (not weekly_uptrend) or sell_breakout or (not squeeze[i]):
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: weekly trend change, buy breakout, or squeeze release
            if (not weekly_downtrend) or buy_breakout or (not squeeze[i]):
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Keltner_Squeeze_Breakout_v1"
timeframe = "1d"
leverage = 1.0