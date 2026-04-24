#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R extreme with 1d EMA34 trend filter and volume confirmation.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d EMA34 for trend filter (price above/below EMA34 defines bull/bear regime).
- Entry: Long when Williams %R(14) < -80 (oversold) in bull regime with volume > 1.8 * 6h volume MA(20);
         Short when Williams %R(14) > -20 (overbought) in bear regime with volume > 1.8 * 6h volume MA(20).
- Exit: ATR trailing stop (2.5 * ATR(14)) or opposite Williams %R extreme.
- Signal size: 0.25 discrete to balance capture and fee control.
- Designed for BTC/ETH: Williams %R identifies exhaustion points, EMA34 filter avoids counter-trend trades,
  volume spike ensures strong participation. Works in bull (buy dips in uptrend) and bear (sell rallies in downtrend).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 6h data for Williams %R calculation and volume
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 6h volume MA(20) for confirmation
    volume_6h = df_6h['volume'].values
    vol_ma_6h = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    vol_ma_6h_aligned = align_htf_to_ltf(prices, df_6h, vol_ma_6h)
    
    # Calculate 6h ATR(14) for trailing stop
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Williams %R(14) on 6h data
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0
    lowest_since_entry = 0
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20, 14, 14)  # EMA34 needs 34, volume MA needs 20, ATR needs 14, Williams %R needs 14
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_6h_aligned[i]) or 
            np.isnan(williams_r[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Volume spike confirmation: 1.8x threshold (balanced to avoid overtrading)
        vol_spike = curr_volume > 1.8 * vol_ma_6h_aligned[i]
        
        # Williams %R extremes
        williams_oversold = williams_r[i] < -80
        williams_overbought = williams_r[i] > -20
        
        # Trend filter: price above/below 1d EMA34
        bull_regime = curr_close > ema_34_1d_aligned[i]
        bear_regime = curr_close < ema_34_1d_aligned[i]
        
        if position == 0:
            # Check for entry signals
            # Long: Williams %R oversold in bull regime with volume spike
            if williams_oversold and bull_regime and vol_spike:
                signals[i] = 0.25
                position = 1
                highest_since_entry = curr_high
            # Short: Williams %R overbought in bear regime with volume spike
            elif williams_overbought and bear_regime and vol_spike:
                signals[i] = -0.25
                position = -1
                lowest_since_entry = curr_low
        elif position == 1:
            # Long position: update highest and check exit conditions
            highest_since_entry = max(highest_since_entry, curr_high)
            # Exit: ATR trailing stop or opposite signal (Williams %R overbought)
            if curr_low <= highest_since_entry - 2.5 * atr[i] or williams_overbought:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: update lowest and check exit conditions
            lowest_since_entry = min(lowest_since_entry, curr_low)
            # Exit: ATR trailing stop or opposite signal (Williams %R oversold)
            if curr_high >= lowest_since_entry + 2.5 * atr[i] or williams_oversold:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_Extreme_1dEMA34_Trend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0