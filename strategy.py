#!/usr/bin/env python3
"""
4h Camarilla H4L4 Breakout + Volume Spike + 1d EMA34 Trend Filter + Chop Filter
Hypothesis: Camarilla H4/L4 levels represent stronger breakout zones than H3/L3. Breakouts above H4 or below L4 with volume confirmation indicate strong institutional participation. 1d EMA34 filter ensures alignment with daily trend. Chop filter (EWMA of |close-open|/ATR) avoids whipsaws in sideways markets. Discrete sizing (0.0, ±0.25) minimizes fee churn. Target: 25-45 trades/year on 4h.
Works in bull markets via breakouts with trend and in bear markets via trend filter (avoids counter-trend entries) and chop filter (avoids false breakouts in range).
"""

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
    
    # Get 1d data for pivot calculation and EMA trend (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate ATR(14) for chop filter and stoploss
    tr1 = pd.Series(high[1:]) - pd.Series(low[1:])
    tr2 = abs(pd.Series(high[1:]) - pd.Series(close[:-1]))
    tr3 = abs(pd.Series(low[1:]) - pd.Series(close[:-1]))
    tr = pd.concat([pd.Series(tr1), pd.Series(tr2), pd.Series(tr3)], axis=1).max(axis=1)
    atr = tr.ewm(span=14, adjust=False, min_periods=14).mean().values
    # Prepend NaN for first element
    atr = np.concatenate([[np.nan], atr[:-1]])
    
    # Calculate chop filter: EWMA of |close-open|/ATR over 20 periods
    body_to_atr = np.abs(close - open_) / atr if 'open_' in locals() else np.abs(close - prices['open'].values) / atr
    open_ = prices['open'].values
    body_to_atr = np.abs(close - open_) / atr
    chop_raw = np.where(~np.isnan(atr) & (atr > 0), body_to_atr, np.nan)
    chop_ewma = pd.Series(chop_raw).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate Camarilla pivots from previous 1d OHLC
    # H4 = close + 1.5*(high-low), L4 = close - 1.5*(high-low)
    prev_close = df_1d['close'].values
    prev_high = df_1d['high'].values
    prev_low = df_1d['low'].values
    rang = prev_high - prev_low
    H4 = prev_close + 1.5 * rang
    L4 = prev_close - 1.5 * rang
    
    # Align Camarilla levels to 4h (use previous day's levels for current day's trading)
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    
    # Calculate volume spike: current volume > 2.0 * 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for volume MA, EMA, ATR, and chop
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(H4_aligned[i]) or np.isnan(L4_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma[i]) or
            np.isnan(atr[i]) or np.isnan(chop_ewma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_open = open_[i]
        curr_volume = volume[i]
        H4_level = H4_aligned[i]
        L4_level = L4_aligned[i]
        ema_trend = ema_34_1d_aligned[i]
        vol_spike = volume_spike[i]
        chop_value = chop_ewma[i]
        
        if position == 0:
            # Look for entry signals
            # Long: price breaks above H4 AND volume spike AND price > 1d EMA34 (uptrend) AND chop < 0.6 (not too choppy)
            long_entry = (curr_close > H4_level) and vol_spike and (curr_close > ema_trend) and (chop_value < 0.6)
            # Short: price breaks below L4 AND volume spike AND price < 1d EMA34 (downtrend) AND chop < 0.6 (not too choppy)
            short_entry = (curr_close < L4_level) and vol_spike and (curr_close < ema_trend) and (chop_value < 0.6)
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: price crosses below L4 (reversal) OR price < 1d EMA34 (trend change) OR chop > 0.8 (too choppy)
            if (curr_close < L4_level) or (curr_close < ema_trend) or (chop_value > 0.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price crosses above H4 (reversal) OR price > 1d EMA34 (trend change) OR chop > 0.8 (too choppy)
            if (curr_close > H4_level) or (curr_close > ema_trend) or (chop_value > 0.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_H4L4_Breakout_VolumeSpike_1dEMA34_Trend_ChopFilter"
timeframe = "4h"
leverage = 1.0