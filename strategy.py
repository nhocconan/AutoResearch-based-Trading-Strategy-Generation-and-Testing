#!/usr/bin/env python3
"""
1d_Camarilla_R1_S1_Breakout_1wTrend_VolumeSpike
Hypothesis: 1d Camarilla R1/S1 breakout with 1w EMA50 trend filter and volume spike confirmation.
Long when price breaks above R1 in 1w uptrend with volume > 2.0x 20-day average.
Short when price breaks below S1 in 1w downtrend with volume > 2.0x 20-day average.
Exit via ATR-based trailing stop (3*ATR from extreme) or re-entry into Donchian(5) range.
Designed for ~15-25 trades/year by requiring strong breakouts, volume confirmation, and trend alignment.
Works in bull/bear markets via 1w EMA50 filter; avoids whipsaws via volume confirmation and ATR stop.
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
    open_ = prices['open'].values  # needed for Camarilla calculation
    
    # Get 1w data for trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate 1w EMA50 for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Camarilla levels for previous day (using prior day's OHLC)
    # We need to shift by 1 to use previous day's data for today's levels
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_open = np.roll(open_, 1)
    # First value will be invalid due to roll, handled by min_periods later
    
    # Camarilla R1, S1, R4, S4 calculations
    # R1 = Close + (High - Low) * 1.1/12
    # S1 = Close - (High - Low) * 1.1/12
    # R4 = Close + (High - Low) * 1.1/2
    # S4 = Close - (High - Low) * 1.1/2
    hl_range = prev_high - prev_low
    R1 = prev_close + hl_range * 1.1 / 12
    S1 = prev_close - hl_range * 1.1 / 12
    R4 = prev_close + hl_range * 1.1 / 2
    S4 = prev_close - hl_range * 1.1 / 2
    
    # Volume regime: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_regime = volume > (2.0 * vol_ma_20)
    
    # ATR for trailing stop (14-period)
    atr_period = 14
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    # Donchian(5) for exit range (tighter than entry)
    lookback_exit = 5
    highest_high_exit = pd.Series(high).rolling(window=lookback_exit, min_periods=lookback_exit).max().values
    lowest_low_exit = pd.Series(low).rolling(window=lookback_exit, min_periods=lookback_exit).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    long_high = 0.0   # highest close since long entry
    short_low = 0.0   # lowest close since short entry
    
    # Start index: need warmup for calculations
    start_idx = max(100, 20, atr_period)  # 20 for volume MA, atr_period for ATR
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN in rolled values or calculated values)
        if (i < 1 or np.isnan(prev_close[i]) or np.isnan(prev_high[i]) or np.isnan(prev_low[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(R1[i]) or np.isnan(S1[i]) or
            np.isnan(R4[i]) or np.isnan(S4[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(atr[i]) or np.isnan(highest_high_exit[i]) or np.isnan(lowest_low_exit[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        ema_trend = ema_50_1w_aligned[i]
        
        if position == 0:
            # Only trade in trending regimes (1w EMA50 filter)
            if close[i] > ema_trend:  # 1w uptrend regime
                # Long: break above R1 with volume spike
                long_signal = (close[i] > R1[i]) and vol_regime[i]
            else:  # 1w downtrend regime
                # Short: break below S1 with volume spike
                short_signal = (close[i] < S1[i]) and vol_regime[i]
            
            if 'long_signal' in locals() and long_signal:
                signals[i] = 0.25
                position = 1
                long_high = close[i]
                # Clean up
                if 'long_signal' in locals(): del long_signal
                if 'short_signal' in locals(): del short_signal
            elif 'short_signal' in locals() and short_signal:
                signals[i] = -0.25
                position = -1
                short_low = close[i]
                # Clean up
                if 'long_signal' in locals(): del long_signal
                if 'short_signal' in locals(): del short_signal
            else:
                signals[i] = 0.0
                # Clean up signal variables
                if 'long_signal' in locals(): del long_signal
                if 'short_signal' in locals(): del short_signal
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Update highest close
            if close[i] > long_high:
                long_high = close[i]
            # Exit conditions: ATR trailing stop OR re-enter Donchian(5) range
            atr_stop = long_high - 3.0 * atr[i]
            range_exit = (close[i] < highest_high_exit[i] and close[i] > lowest_low_exit[i])
            if close[i] <= atr_stop or range_exit:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Update lowest close
            if close[i] < short_low:
                short_low = close[i]
            # Exit conditions: ATR trailing stop OR re-enter Donchian(5) range
            atr_stop = short_low + 3.0 * atr[i]
            range_exit = (close[i] > lowest_low_exit[i] and close[i] < highest_high_exit[i])
            if close[i] >= atr_stop or range_exit:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Camarilla_R1_S1_Breakout_1wTrend_VolumeSpike"
timeframe = "1d"
leverage = 1.0