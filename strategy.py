#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA34 trend filter and volume confirmation
# Uses weekly EMA34 for long-term trend direction (works in both bull/bear markets)
# Donchian breakout provides structure with clear entry/exit levels
# Volume confirmation ensures participation and reduces false breakouts
# ATR-based stoploss manages risk without look-ahead
# Discrete position sizing 0.25 balances risk and minimizes fee churn
# Targets 7-25 trades/year (30-100 total over 4 years) to stay within fee drag limits

name = "1d_Donchian20_1wEMA34_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 35:
        return np.zeros(n)
    
    # Calculate weekly EMA34 for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align weekly EMA to daily
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate daily Donchian channels (20-period)
    # Donchian Upper = max(high, lookback=20)
    # Donchian Lower = min(low, lookback=20)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Calculate daily ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with close index
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate volume confirmation (2x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_confirm = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for Donchian and ATR)
    start_idx = 34  # max(20 for Donchian, 14 for ATR) + buffer
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(atr_14[i]) or 
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above Donchian Upper AND above weekly EMA34 AND volume confirm
            if (close[i] > donchian_upper[i] and 
                close[i] > ema_34_1w_aligned[i] and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian Lower AND below weekly EMA34 AND volume confirm
            elif (close[i] < donchian_lower[i] and 
                  close[i] < ema_34_1w_aligned[i] and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Stoploss: price drops below entry - 2*ATR (using ATR from 2 bars ago to avoid look-ahead)
            atr_stop = atr_14[i-2] if i >= 2 and not np.isnan(atr_14[i-2]) else 0
            if atr_stop > 0 and close[i] < (donchian_upper[i-1] - 2 * atr_stop):
                signals[i] = 0.0
                position = 0
            # Exit: price breaks below Donchian Lower
            elif close[i] < donchian_lower[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: price rises above entry + 2*ATR (using ATR from 2 bars ago)
            atr_stop = atr_14[i-2] if i >= 2 and not np.isnan(atr_14[i-2]) else 0
            if atr_stop > 0 and close[i] > (donchian_lower[i-1] + 2 * atr_stop):
                signals[i] = 0.0
                position = 0
            # Exit: price breaks above Donchian Upper
            elif close[i] > donchian_upper[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals