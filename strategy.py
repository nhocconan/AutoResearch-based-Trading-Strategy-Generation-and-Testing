#!/usr/bin/env python3
"""
Hypothesis: 12h Williams %R extreme with 1d EMA34 trend filter and volume spike.
- Primary timeframe: 12h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d EMA34 for trend filter (price above/below EMA34 defines bull/bear regime).
- Entry: Long when Williams %R(14) crosses above -20 in bull regime with volume > 1.8 * 12h volume MA(20);
         Short when Williams %R(14) crosses below -80 in bear regime with volume > 1.8 * 12h volume MA(20).
- Exit: ATR trailing stop (2.0 * ATR(14)) or opposite Williams %R extreme.
- Signal size: 0.25 discrete to balance capture and fee control.
- Designed for BTC/ETH: Williams %R identifies overbought/oversold extremes, EMA34 filter avoids counter-trend trades,
  volume spike ensures strong participation. Works in bull (pullbacks to extreme in uptrend) and bear (bounces from extreme in downtrend).
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
    
    # Get 12h data for Williams %R calculation and volume
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 12h volume MA(20) for confirmation
    volume_12h = df_12h['volume'].values
    vol_ma_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    
    # Calculate 12h ATR(14) for trailing stop
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Williams %R(14) on 12h data
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    # Handle division by zero (when highest_high == lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20, 14)  # EMA34 needs 34, volume MA needs 20, ATR needs 14, Williams %R needs 14
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_12h_aligned[i]) or 
            np.isnan(atr[i]) or np.isnan(williams_r[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Volume spike confirmation: 1.8x threshold (tight to reduce trades)
        vol_spike = curr_volume > 1.8 * vol_ma_12h_aligned[i]
        
        # Trend filter: price above/below 1d EMA34
        bull_regime = curr_close > ema_34_1d_aligned[i]
        bear_regime = curr_close < ema_34_1d_aligned[i]
        
        if position == 0:
            # Check for entry signals
            # Long: Williams %R crosses above -20 (exiting oversold) in bull regime with volume spike
            if williams_r[i] > -20 and williams_r[i-1] <= -20 and bull_regime and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -80 (exiting overbought) in bear regime with volume spike
            elif williams_r[i] < -80 and williams_r[i-1] >= -80 and bear_regime and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: check exit conditions
            # Exit: ATR trailing stop or Williams %R crosses below -80 (overbought)
            if low[i] <= high[i-1] - 2.0 * atr[i] or williams_r[i] < -80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: check exit conditions
            # Exit: ATR trailing stop or Williams %R crosses above -20 (oversold)
            if high[i] >= low[i-1] + 2.0 * atr[i] or williams_r[i] > -20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsR_Extreme_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0