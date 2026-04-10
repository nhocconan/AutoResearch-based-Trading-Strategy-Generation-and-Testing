#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Elder Ray Index with 1d ADX regime filter and volume confirmation
# - Long when Elder Ray Bull Power > 0 AND Bear Power < 0 AND 1d ADX > 25 AND volume > 1.5x 20-period average volume
# - Short when Elder Ray Bear Power < 0 AND Bull Power > 0 AND 1d ADX > 25 AND volume > 1.5x 20-period average volume
# - Exit when Elder Ray Bull Power < 0 (for longs) or Bear Power > 0 (for shorts) OR ADX < 20 (regime change)
# - Uses discrete position sizing 0.25 to limit fee churn
# - Target: 12-37 trades/year on 12h timeframe (50-150 total over 4 years)
# - Elder Ray measures bull/bear power relative to EMA13, effective in both trending and ranging markets
# - ADX filter ensures we trade only when trend strength is sufficient (>25) and exits when weak (<20) to avoid whipsaws
# - Volume confirmation reduces false signals

name = "12h_1d_elder_ray_adx_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 12h OHLC and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute 12h EMA13 for Elder Ray
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Pre-compute 12h Elder Ray Index
    bull_power = high - ema13  # Bull Power = High - EMA13
    bear_power = low - ema13   # Bear Power = Low - EMA13
    
    # Pre-compute 12h volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Pre-compute 1d ADX for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0  # First bar has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(14) using Wilder's smoothing
    atr_1d = np.zeros_like(tr)
    atr_1d[13] = np.mean(tr[1:14])  # First ATR value
    for i in range(14, len(tr)):
        atr_1d[i] = (atr_1d[i-1] * 13 + tr[i]) / 14
    
    # Directional Movement calculation
    plus_dm = np.zeros_like(high_1d)
    minus_dm = np.zeros_like(high_1d)
    plus_dm[0] = 0
    minus_dm[0] = 0
    for i in range(1, len(high_1d)):
        up_move = high_1d[i] - high_1d[i-1]
        down_move = low_1d[i-1] - low_1d[i]
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        else:
            plus_dm[i] = 0
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
        else:
            minus_dm[i] = 0
    
    # Smoothed Directional Movement (+DI and -DI)
    atr_period = 14
    smoothed_plus_dm = np.zeros_like(plus_dm)
    smoothed_minus_dm = np.zeros_like(minus_dm)
    smoothed_tr = np.zeros_like(tr)
    
    # Initial values
    smoothed_plus_dm[atr_period] = np.sum(plus_dm[1:atr_period+1])
    smoothed_minus_dm[atr_period] = np.sum(minus_dm[1:atr_period+1])
    smoothed_tr[atr_period] = np.sum(tr[1:atr_period+1])
    
    # Wilder's smoothing
    for i in range(atr_period+1, len(high_1d)):
        smoothed_plus_dm[i] = (smoothed_plus_dm[i-1] * (atr_period-1) + plus_dm[i]) / atr_period
        smoothed_minus_dm[i] = (smoothed_minus_dm[i-1] * (atr_period-1) + minus_dm[i]) / atr_period
        smoothed_tr[i] = (smoothed_tr[i-1] * (atr_period-1) + tr[i]) / atr_period
    
    # Calculate +DI and -DI
    plus_di = np.zeros_like(high_1d)
    minus_di = np.zeros_like(high_1d)
    for i in range(atr_period, len(high_1d)):
        if smoothed_tr[i] != 0:
            plus_di[i] = (smoothed_plus_dm[i] / smoothed_tr[i]) * 100
            minus_di[i] = (smoothed_minus_dm[i] / smoothed_tr[i]) * 100
    
    # Calculate DX and ADX
    dx = np.zeros_like(high_1d)
    for i in range(atr_period, len(high_1d)):
        if (plus_di[i] + minus_di[i]) != 0:
            dx[i] = (np.abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])) * 100
    
    adx_1d = np.zeros_like(high_1d)
    adx_1d[2*atr_period-1] = np.mean(dx[atr_period:2*atr_period])  # First ADX value
    for i in range(2*atr_period, len(high_1d)):
        adx_1d[i] = (adx_1d[i-1] * (atr_period-1) + dx[i]) / atr_period
    
    # ADX regime: strong trend when ADX > 25, weak trend when ADX < 20
    strong_trend = adx_1d > 25
    weak_trend = adx_1d < 20
    
    # Align HTF indicators to 12h timeframe
    strong_trend_aligned = align_htf_to_ltf(prices, df_1d, strong_trend)
    weak_trend_aligned = align_htf_to_ltf(prices, df_1d, weak_trend)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(ema13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(strong_trend_aligned[i]) or np.isnan(weak_trend_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: Bull Power > 0 AND Bear Power < 0 AND strong trend AND volume spike
            if (bull_power[i] > 0 and 
                bear_power[i] < 0 and 
                strong_trend_aligned[i] and 
                volume_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short conditions: Bear Power < 0 AND Bull Power > 0 AND strong trend AND volume spike
            elif (bear_power[i] < 0 and 
                  bull_power[i] > 0 and 
                  strong_trend_aligned[i] and 
                  volume_spike[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: 
            # For longs: Bull Power < 0 OR weak trend regime
            # For shorts: Bear Power > 0 OR weak trend regime
            exit_long = (position == 1 and (bull_power[i] < 0 or weak_trend_aligned[i]))
            exit_short = (position == -1 and (bear_power[i] > 0 or weak_trend_aligned[i]))
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals