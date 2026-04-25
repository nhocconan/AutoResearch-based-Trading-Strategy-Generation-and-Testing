#!/usr/bin/env python3
"""
6h Elder Ray + ADX Regime + Volume Spike
Hypothesis: Elder Ray (Bull/Bear Power) identifies bull/bear momentum via EMA13, ADX confirms regime strength (>25), and volume spike filters for institutional participation. Works in bull (long when Bull Power > 0, ADX > 25, volume spike) and bear (short when Bear Power < 0, ADX > 25, volume spike). Uses 1d EMA for HTF trend alignment to avoid counter-trend trades. Target 12-37 trades/year on 6h to stay within fee drag limits.
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
    
    # Get 1d data for EMA34 trend filter (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = pd.Series(df_1d['close'])
    ema_34_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate EMA13 for Elder Ray (primary timeframe)
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate True Range and ATR(14) for ADX
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr = np.full(n, np.nan)
    for i in range(14, n):
        atr[i] = np.mean(tr[i-13:i+1])
    
    # Calculate +DM and -DM for ADX
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    for i in range(1, n):
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        else:
            plus_dm[i] = 0
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
        else:
            minus_dm[i] = 0
    
    # Calculate smoothed +DM, -DM, and TR for ADX (using Wilder's smoothing)
    def wilders_smoothing(values, period):
        result = np.full_like(values, np.nan)
        if len(values) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(values[0:period])
        # Subsequent values: Wilder's smoothing
        for i in range(period, len(values)):
            result[i] = (result[i-1] * (period-1) + values[i]) / period
        return result
    
    plus_dm_smooth = wilders_smoothing(plus_dm, 14)
    minus_dm_smooth = wilders_smoothing(minus_dm, 14)
    tr_smooth = wilders_smoothing(tr, 14)
    
    # Calculate +DI and -DI
    plus_di = np.full(n, np.nan)
    minus_di = np.full(n, np.nan)
    for i in range(14, n):
        if tr_smooth[i] != 0:
            plus_di[i] = (plus_dm_smooth[i] / tr_smooth[i]) * 100
            minus_di[i] = (minus_dm_smooth[i] / tr_smooth[i]) * 100
    
    # Calculate DX and ADX
    dx = np.full(n, np.nan)
    for i in range(14, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum != 0:
            dx[i] = abs(plus_di[i] - minus_di[i]) / di_sum * 100
    adx = wilders_smoothing(dx, 14)
    
    # Calculate Elder Ray components
    bull_power = high - ema13  # Bull Power: High - EMA13
    bear_power = low - ema13   # Bear Power: Low - EMA13
    
    # Calculate 20-period volume MA for volume confirmation
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for all indicators
    start_idx = max(34, 13, 14*3, 20)  # EMA34, EMA13, ADX (14*3 for smoothing), volume MA
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(adx[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        ema_34_val = ema_34_1d_aligned[i]
        adx_val = adx[i]
        bull_power_val = bull_power[i]
        bear_power_val = bear_power[i]
        vol_ma = vol_ma_20[i]
        curr_volume = volume[i]
        
        # Trend filter: price relative to 1d EMA34 (avoid counter-trend trades)
        uptrend = curr_close > ema_34_val
        downtrend = curr_close < ema_34_val
        
        # ADX filter: regime strength (>25 = trending market)
        strong_trend = adx_val > 25
        
        # Volume confirmation: current volume > 2.0 * 20-period average
        volume_confirm = curr_volume > 2.0 * vol_ma
        
        if position == 0:
            # Long: Bull Power > 0 (bulls in control) + strong trend + volume spike + uptrend
            long_signal = (bull_power_val > 0) and strong_trend and volume_confirm and uptrend
            # Short: Bear Power < 0 (bears in control) + strong trend + volume spike + downtrend
            short_signal = (bear_power_val < 0) and strong_trend and volume_confirm and downtrend
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management: exit when Bull Power turns negative OR trend weakens
            if bull_power_val <= 0 or adx_val < 20 or curr_close < ema_34_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management: exit when Bear Power turns positive OR trend weakens
            if bear_power_val >= 0 or adx_val < 20 or curr_close > ema_34_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_ADX_Regime_VolumeSpike"
timeframe = "6h"
leverage = 1.0