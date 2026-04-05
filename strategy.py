#!/usr/bin/env python3
"""
Experiment #8114: 1-hour momentum with 4h trend filter and 1d regime filter.
Hypothesis: On 1h, buy when price crosses above 20-period EMA with volume confirmation 
and 4h trend is bullish (price > 4h EMA50) and 1d regime is not extreme (ADX < 30); 
sell when price crosses below 20-period EMA with volume confirmation and 4h trend bearish 
(price < 4h EMA50) or 1d regime is trending (ADX > 30). Uses 4h for trend direction, 
1d for regime filter (avoid whipsaw in strong trends), and 1h for precise entry/exit.
Targets 15-37 trades/year by requiring multiple confluence factors.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_8114_1h_momentum_4h_trend_1d_regime_v1"
timeframe = "1h"
leverage = 1.0

# Parameters
EMA_FAST = 20
EMA_TREND = 50
VOLUME_MA = 20
VOLUME_THRESHOLD = 1.5
ADX_PERIOD = 14
ADX_TREND_THRESHOLD = 30
SIGNAL_SIZE = 0.20
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # 4h EMA for trend direction
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=EMA_TREND, adjust=False, min_periods=EMA_TREND).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # 1d ADX for regime detection
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = pd.Series(high_1d - low_1d)
    tr2 = pd.Series(np.abs(high_1d - np.roll(close_1d, 1)))
    tr3 = pd.Series(np.abs(low_1d - np.roll(close_1d, 1)))
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    dm_plus = pd.Series(np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                                 np.maximum(high_1d - np.roll(high_1d, 1), 0), 0))
    dm_minus = pd.Series(np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                                  np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0))
    
    # Smoothed values
    atr_1d = tr_1d.ewm(span=ADX_PERIOD, adjust=False, min_periods=ADX_PERIOD).mean().values
    dm_plus_smooth = dm_plus.ewm(span=ADX_PERIOD, adjust=False, min_periods=ADX_PERIOD).mean().values
    dm_minus_smooth = dm_minus.ewm(span=ADX_PERIOD, adjust=False, min_periods=ADX_PERIOD).mean().values
    
    # Directional Indicators
    plus_di = 100 * dm_plus_smooth / atr_1d
    minus_di = 100 * dm_minus_smooth / atr_1d
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(span=ADX_PERIOD, adjust=False, min_periods=ADX_PERIOD).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Fast EMA for entry signal
    ema_fast = pd.Series(close).ewm(span=EMA_FAST, adjust=False, min_periods=EMA_FAST).mean().values
    
    # Volume moving average
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA, min_periods=VOLUME_MA).mean().values
    
    # ATR for risk management
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(EMA_FAST, EMA_TREND, VOLUME_MA, ATR_PERIOD, ADX_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(ema_4h_aligned[i]) or np.isnan(adx_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Check stoploss
        if position == 1 and close[i] <= stop_price:
            signals[i] = 0.0
            position = 0
            continue
        elif position == -1 and close[i] >= stop_price:
            signals[i] = 0.0
            position = 0
            continue
        
        # Determine conditions
        price_above_ema_fast = close[i] > ema_fast[i]
        price_below_ema_fast = close[i] < ema_fast[i]
        
        # 4h trend: bullish if price > EMA50, bearish if price < EMA50
        bullish_4h = close[i] > ema_4h_aligned[i]
        bearish_4h = close[i] < ema_4h_aligned[i]
        
        # 1d regime: ranging if ADX < 30, trending if ADX >= 30
        ranging_1d = adx_aligned[i] < ADX_TREND_THRESHOLD
        trending_1d = adx_aligned[i] >= ADX_TREND_THRESHOLD
        
        # Volume confirmation
        volume_confirmed = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Entry logic
        if position == 0:
            # Long: price crosses above fast EMA, 4h bullish, 1d ranging, volume confirmation
            if price_above_ema_fast and bullish_4h and ranging_1d and volume_confirmed:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            # Short: price crosses below fast EMA, 4h bearish, 1d ranging, volume confirmation
            elif price_below_ema_fast and bearish_4h and ranging_1d and volume_confirmed:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price crosses below fast EMA OR 4h turns bearish OR 1d becomes trending
            if price_below_ema_fast or not bullish_4h or trending_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            # Short exit: price crosses above fast EMA OR 4h turns bullish OR 1d becomes trending
            if price_above_ema_fast or not bearish_4h or trending_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals