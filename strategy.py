# Hypothesis: In strong trends (ADX > 25), price tends to retrace to the 20-period EMA before continuing. 
# We use 4h chart for entry timing, with 1d ADX for trend strength filter and 1d EMA20 as dynamic support/resistance.
# Enter long when price touches EMA20 in uptrend (ADX > 25 and price > EMA50), short when touches EMA20 in downtrend.
# Volume confirmation filters out weak touches. Stop when trend weakens (ADX < 20) or price breaks EMA50.
# This mean-reversion-within-trend approach works in both bull and bear markets by capturing retracements.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data once for ADX and EMAs
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 30:
        return np.zeros(n)
    
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    close_daily = df_daily['close'].values
    
    # Daily ADX (14) for trend strength
    # True Range
    tr1 = np.abs(high_daily - low_daily)
    tr2 = np.abs(high_daily - np.roll(close_daily, 1))
    tr3 = np.abs(low_daily - np.roll(close_daily, 1))
    tr1[0] = high_daily[0] - low_daily[0]
    tr2[0] = np.abs(high_daily[0] - close_daily[0])
    tr3[0] = np.abs(low_daily[0] - close_daily[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = np.zeros_like(high_daily)
    down_move = np.zeros_like(low_daily)
    up_move[1:] = np.maximum(high_daily[1:] - high_daily[:-1], 0)
    down_move[1:] = np.maximum(low_daily[:-1] - low_daily[1:], 0)
    
    # Smoothed values
    tr_ma = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean()
    plus_dm = pd.Series(up_move).ewm(alpha=1/14, adjust=False).mean()
    minus_dm = pd.Series(down_move).ewm(alpha=1/14, adjust=False).mean()
    
    # DI and DX
    plus_di = 100 * plus_dm / tr_ma
    minus_di = 100 * minus_dm / tr_ma
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    
    # Daily EMA20 (dynamic support/resistance) and EMA50 (trend filter)
    ema20_daily = pd.Series(close_daily).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema50_daily = pd.Series(close_daily).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF indicators to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_daily, adx)
    ema20_daily_aligned = align_htf_to_ltf(prices, df_daily, ema20_daily)
    ema50_daily_aligned = align_htf_to_ltf(prices, df_daily, ema50_daily)
    
    # Main timeframe data (4h)
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if NaN in critical values
        if (np.isnan(adx_aligned[i]) or np.isnan(ema20_daily_aligned[i]) or 
            np.isnan(ema50_daily_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        adx_val = adx_aligned[i]
        ema20 = ema20_daily_aligned[i]
        ema50 = ema50_daily_aligned[i]
        vol_current = volume[i]
        
        # Volume filter: current volume > 1.5x 20-period average
        vol_ma = np.mean(volume[max(0, i-20):i]) if i >= 20 else volume[i]
        vol_ok = vol_current > 1.5 * vol_ma
        
        # Trend and entry conditions
        uptrend = adx_val > 25 and price > ema50
        downtrend = adx_val > 25 and price < ema50
        
        # Price near EMA20 (within 1%)
        near_ema20 = np.abs(price - ema20) / ema20 < 0.01
        
        if position == 0:
            # Long: price touches EMA20 in uptrend with volume
            if uptrend and near_ema20 and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short: price touches EMA20 in downtrend with volume
            elif downtrend and near_ema20 and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: trend weakens or price breaks below EMA50
            if adx_val < 20 or price < ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: trend weakens or price breaks above EMA50
            if adx_val < 20 or price > ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_1d_EMA20_Retracement_ADX_VolumeFilter_v1"
timeframe = "4h"
leverage = 1.0