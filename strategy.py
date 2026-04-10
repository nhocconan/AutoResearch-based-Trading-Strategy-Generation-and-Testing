#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + ADX regime filter + volume confirmation
# - Elder Ray: Bull Power = High - EMA(13), Bear Power = EMA(13) - Low
# - Trend regime: ADX(14) > 25 indicates trending market
# - Long when Bull Power > 0 AND ADX > 25 AND volume > 1.5x 20-period average
# - Short when Bear Power > 0 AND ADX > 25 AND volume > 1.5x 20-period average
# - Exit when Elder Ray power reverses sign OR ADX < 20 (regime change to ranging)
# - Uses discrete position sizing 0.25 to limit fee churn
# - Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years)
# - Elder Ray captures trend strength via price position relative to EMA
# - ADX filter ensures we only trade in trending markets where Elder Ray works best
# - Volume confirmation reduces false signals

name = "6h_1w_elder_ray_adx_volume_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Pre-compute 6h OHLC and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute 6h EMA(13) for Elder Ray
    close_s = pd.Series(close)
    ema_13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema_13  # Higher highs relative to trend
    bear_power = ema_13 - low   # Lower lows relative to trend
    
    # Pre-compute 6h volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Pre-compute 6h ADX(14) for regime filter
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    up_move[0] = 0
    down_move[0] = 0
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed TR, +DM, -DM using Wilder's smoothing (alpha=1/14)
    def wilder_smoothing(data, period):
        result = np.zeros_like(data)
        result[period-1] = np.mean(data[1:period])  # First value
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr_14 = wilder_smoothing(tr, 14)
    plus_dm_14 = wilder_smoothing(plus_dm, 14)
    minus_dm_14 = wilder_smoothing(minus_dm, 14)
    
    # Directional Indicators
    plus_di_14 = 100 * plus_dm_14 / np.where(atr_14 == 0, 1, atr_14)
    minus_di_14 = 100 * minus_dm_14 / np.where(atr_14 == 0, 1, atr_14)
    
    # ADX calculation
    dx = 100 * np.abs(plus_di_14 - minus_di_14) / np.where((plus_di_14 + minus_di_14) == 0, 1, (plus_di_14 + minus_di_14))
    adx = np.zeros_like(dx)
    adx[13] = np.mean(dx[1:14])  # First ADX value
    for i in range(14, len(dx)):
        adx[i] = (adx[i-1] * 13 + dx[i]) / 14
    
    # ADX regime: trending when ADX > 25
    trending_regime = adx > 25
    
    # Align HTF indicators to 6h timeframe (1w trend filter)
    # Use weekly close price relative to weekly EMA as trend filter
    weekly_close = df_1w['close'].values
    weekly_ema_21 = pd.Series(weekly_close).ewm(span=21, adjust=False, min_periods=21).mean().values
    weekly_uptrend = weekly_close > weekly_ema_21
    weekly_uptrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_uptrend)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(ema_13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(adx[i]) or np.isnan(weekly_uptrend_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: Bull Power > 0 AND trending regime AND volume spike AND weekly uptrend
            if (bull_power[i] > 0 and 
                trending_regime[i] and 
                volume_spike[i] and 
                weekly_uptrend_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short conditions: Bear Power > 0 AND trending regime AND volume spike AND weekly downtrend
            elif (bear_power[i] > 0 and 
                  trending_regime[i] and 
                  volume_spike[i] and 
                  not weekly_uptrend_aligned[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: Elder Ray power reverses OR ADX < 20 (ranging) OR weekly trend change
            exit_long = (position == 1 and 
                        (bull_power[i] <= 0 or adx[i] < 20 or not weekly_uptrend_aligned[i]))
            exit_short = (position == -1 and 
                         (bear_power[i] <= 0 or adx[i] < 20 or weekly_uptrend_aligned[i]))
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals