#!/usr/bin/env python3
"""
Hypothesis: 12h Williams %R mean reversion with 1d trend filter and volume confirmation.
Long when Williams %R < -80 (oversold) AND 1d close > 1d EMA34 AND 12h volume > 1.8x 20-period average volume.
Short when Williams %R > -20 (overbought) AND 1d close < 1d EMA34 AND 12h volume > 1.8x 20-period average volume.
Exit when Williams %R crosses above -50 (for longs) or below -50 (for shorts) OR ATR trailing stop (2.0*ATR from extreme).
Williams %R identifies overextended moves; mean reversion works in both trending and ranging markets when filtered by 1d trend and volume spikes.
Target: 12-30 trades/year on 12h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # Need enough for EMA34
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Williams %R(14) = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low + 1e-10) * -100
    
    # 12h volume average (20-period) for spike filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for 12h trailing stop calculation
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0  # for long trailing stop
    lowest_since_entry = 0.0   # for short trailing stop
    
    # Start from index where all indicators are ready
    start_idx = max(20, 14, 34)  # vol_ma20, williams_r14, ema34_1d
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr[i]) or np.isnan(williams_r[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        ema_val = ema_34_1d_aligned[i]
        wr = williams_r[i]
        
        if position == 0:
            # Long: Williams %R < -80 (oversold) AND bullish trend (1d close > EMA34) AND volume spike
            if wr < -80 and close[i] > ema_val and volume[i] > 1.8 * vol_ma_val:
                signals[i] = 0.25
                position = 1
                highest_since_entry = price
            # Short: Williams %R > -20 (overbought) AND bearish trend (1d close < EMA34) AND volume spike
            elif wr > -20 and close[i] < ema_val and volume[i] > 1.8 * vol_ma_val:
                signals[i] = -0.25
                position = -1
                lowest_since_entry = price
        else:
            # Update highest/lowest since entry for trailing stop
            if position == 1:
                highest_since_entry = max(highest_since_entry, price)
            elif position == -1:
                lowest_since_entry = min(lowest_since_entry, price)
            
            # Exit conditions
            exit_signal = False
            
            # Primary exit: Williams %R crosses above -50 (for longs) or below -50 (for shorts)
            if position == 1 and wr > -50:
                exit_signal = True
            elif position == -1 and wr < -50:
                exit_signal = True
            
            # ATR-based trailing stop: 2.0 * ATR from highest/lowest since entry
            if position == 1 and price < highest_since_entry - 2.0 * atr_val:
                exit_signal = True
            elif position == -1 and price > lowest_since_entry + 2.0 * atr_val:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_WilliamsR_MeanReversion_1dEMA34_Trend_VolumeSpike_WR50Exit_ATRTrailingStop"
timeframe = "12h"
leverage = 1.0