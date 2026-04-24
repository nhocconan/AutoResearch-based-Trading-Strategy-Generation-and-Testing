#!/usr/bin/env python3
"""
Hypothesis: 1d Williams %R extreme reversal with 1w EMA34 trend filter and volume spike.
- Primary timeframe: 1d targeting 30-100 total trades over 4 years (7-25/year).
- HTF: 1w EMA34 for trend filter (price above/below EMA34 defines bull/bear regime).
- Entry: Long when Williams %R(14) crosses above -80 from below in bull regime with volume > 1.5 * 1d volume MA(20);
         Short when Williams %R(14) crosses below -20 from above in bear regime with volume > 1.5 * 1d volume MA(20).
- Exit: ATR trailing stop (2.0 * ATR(14)) or opposite Williams %R extreme (above -20 for long, below -80 for short).
- Signal size: 0.25 discrete to balance capture and fee control.
- Designed for BTC/ETH: Williams %R captures overextended moves in both bull and bear markets,
  1w EMA34 filter ensures trading with the higher timeframe trend, volume spike confirms participation.
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
    
    # Get 1d data for Williams %R calculation and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Get 1w data for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 1w EMA34 for trend filter
    ema_34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate 1d volume MA(20) for confirmation
    volume_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Calculate 1d ATR(14) for trailing stop
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Williams %R(14) on 1d data
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    # Handle division by zero (when highest_high == lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0
    lowest_since_entry = 0
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20, 14) + 1  # EMA34 needs 34, volume MA needs 20, ATR needs 14, plus 1 for safety
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]) or 
            np.isnan(williams_r[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_williams_r = williams_r[i]
        prev_williams_r = williams_r[i-1] if i > 0 else -50
        
        # Volume spike confirmation: 1.5x threshold (tight to reduce trades)
        vol_spike = curr_volume > 1.5 * vol_ma_1d_aligned[i]
        
        # Trend filter: price above/below 1w EMA34
        bull_regime = curr_close > ema_34_1w_aligned[i]
        bear_regime = curr_close < ema_34_1w_aligned[i]
        
        # Williams %R conditions
        williams_r_oversold = curr_williams_r < -80  # Oversold condition
        williams_r_overbought = curr_williams_r > -20  # Overbought condition
        williams_r_cross_above_80 = prev_williams_r <= -80 and curr_williams_r > -80  # Cross above -80
        williams_r_cross_below_20 = prev_williams_r >= -20 and curr_williams_r < -20  # Cross below -20
        
        if position == 0:
            # Check for entry signals
            # Long: Williams %R crosses above -80 from below in bull regime with volume spike
            if williams_r_cross_above_80 and bull_regime and vol_spike:
                signals[i] = 0.25
                position = 1
                highest_since_entry = curr_high
            # Short: Williams %R crosses below -20 from above in bear regime with volume spike
            elif williams_r_cross_below_20 and bear_regime and vol_spike:
                signals[i] = -0.25
                position = -1
                lowest_since_entry = curr_low
        elif position == 1:
            # Long position: update highest and check exit conditions
            highest_since_entry = max(highest_since_entry, curr_high)
            # Exit: ATR trailing stop or opposite condition (Williams %R above -20)
            if curr_low <= highest_since_entry - 2.0 * atr[i] or curr_williams_r > -20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: update lowest and check exit conditions
            lowest_since_entry = min(lowest_since_entry, curr_low)
            # Exit: ATR trailing stop or opposite condition (Williams %R below -80)
            if curr_high >= lowest_since_entry + 2.0 * atr[i] or curr_williams_r < -80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WilliamsR_Extreme_1wEMA34_Trend_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0