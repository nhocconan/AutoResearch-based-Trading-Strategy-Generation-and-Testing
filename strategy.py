#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 12h ADX regime filter + volume confirmation
# - Elder Ray: Bull Power = High - EMA13, Bear Power = EMA13 - Low
# - Long when Bull Power > 0 AND Bear Power increasing (less negative) AND 12h ADX > 25 (trending regime) AND volume > 1.5x 20-period average
# - Short when Bear Power < 0 AND Bull Power decreasing (less positive) AND 12h ADX > 25 AND volume > 1.5x 20-period average
# - Exit when Elder Ray power diverges (Bull Power < 0 for long, Bear Power > 0 for short) OR ADX < 20 (regime change)
# - Uses discrete position sizing 0.25 to limit fee churn
# - Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years)
# - Elder Ray measures bull/bear power relative to EMA13, effective in both trending and ranging markets
# - ADX filter ensures we trade only when trend is strong enough to follow
# - Volume confirmation reduces false signals

name = "6h_12h_elder_ray_adx_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Pre-compute 6h OHLC and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute 6h EMA13 for Elder Ray
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=13, min_periods=13, adjust=False).mean().values
    
    # Pre-compute 6h Elder Ray components
    bull_power = high - ema13  # Bull Power: High - EMA13
    bear_power = ema13 - low   # Bear Power: EMA13 - Low
    
    # Pre-compute 6b volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Pre-compute 12h ADX for regime filter
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range calculation
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr1[0] = 0  # First bar has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(14) using Wilder's smoothing
    atr_12h = np.zeros_like(tr)
    atr_12h[13] = np.mean(tr[1:14])  # First ATR value
    for i in range(14, len(tr)):
        atr_12h[i] = (atr_12h[i-1] * 13 + tr[i]) / 14
    
    # +DM and -DM calculation
    up_move = high_12h - np.roll(high_12h, 1)
    down_move = np.roll(low_12h, 1) - low_12h
    up_move[0] = 0
    down_move[0] = 0
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed +DM, -DM, and ATR
    tr_period = 14
    plus_dm_smooth = np.zeros_like(plus_dm)
    minus_dm_smooth = np.zeros_like(minus_dm)
    atr_smooth = np.zeros_like(atr_12h)
    
    # Initial values
    plus_dm_smooth[tr_period] = np.sum(plus_dm[1:tr_period+1])
    minus_dm_smooth[tr_period] = np.sum(minus_dm[1:tr_period+1])
    atr_smooth[tr_period] = atr_12h[tr_period]
    
    # Wilder's smoothing
    for i in range(tr_period+1, len(tr)):
        plus_dm_smooth[i] = plus_dm_smooth[i-1] - (plus_dm_smooth[i-1] / tr_period) + plus_dm[i]
        minus_dm_smooth[i] = minus_dm_smooth[i-1] - (minus_dm_smooth[i-1] / tr_period) + minus_dm[i]
        atr_smooth[i] = atr_smooth[i-1] - (atr_smooth[i-1] / tr_period) + atr_12h[i]
    
    # +DI and -DI
    plus_di = np.where(atr_smooth != 0, (plus_dm_smooth / atr_smooth) * 100, 0)
    minus_di = np.where(atr_smooth != 0, (minus_dm_smooth / atr_smooth) * 100, 0)
    
    # DX and ADX
    dx = np.where((plus_di + minus_di) != 0, np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100, 0)
    adx_12h = np.zeros_like(dx)
    adx_12h[tr_period*2] = np.mean(dx[tr_period:tr_period*2+1])  # First ADX value
    for i in range(tr_period*2+1, len(dx)):
        adx_12h[i] = (adx_12h[i-1] * (tr_period-1) + dx[i]) / tr_period
    
    # ADX regime: trending when ADX > 25
    trending_regime = adx_12h > 25
    
    # Align HTF indicators to 6h timeframe
    trending_regime_aligned = align_htf_to_ltf(prices, df_12h, trending_regime)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(trending_regime_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: Bull Power > 0 AND Bear Power increasing (less negative) AND trending regime AND volume spike
            if (bull_power[i] > 0 and 
                bear_power[i] > bear_power[i-1] and  # Bear Power increasing (less negative)
                trending_regime_aligned[i] and 
                volume_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short conditions: Bear Power < 0 AND Bull Power decreasing (less positive) AND trending regime AND volume spike
            elif (bear_power[i] < 0 and 
                  bull_power[i] < bull_power[i-1] and  # Bull Power decreasing (less positive)
                  trending_regime_aligned[i] and 
                  volume_spike[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: Elder Ray power diverges OR regime change (ADX < 20)
            exit_long = (position == 1 and (bull_power[i] < 0 or not trending_regime_aligned[i]))
            exit_short = (position == -1 and (bear_power[i] > 0 or not trending_regime_aligned[i]))
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals