#!/usr/bin/env python3
"""
Experiment #8834: 1h mean reversion with 4h/1d trend filter and volume confirmation.
Hypothesis: In 1h timeframe, mean reversion works better during ranging markets when aligned with higher timeframe trend.
Use 4h for trend direction (EMA50), 1d for regime filter (ADX < 25 = range), and 1h for entry (RSI extremes + Bollinger Bands).
Volume confirmation filters false signals. Targets 60-150 trades over 4 years to minimize fee drag.
Works in both bull/bear: range trading in sideways markets, trend alignment avoids counter-trend trades.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_8834_1h_meanrev_4h_trend_1d_regime_vol_v1"
timeframe = "1h"
leverage = 1.0

# Parameters
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
BB_PERIOD = 20
BB_STD = 2.0
ADX_PERIOD = 14
ADX_THRESHOLD = 25  # Below = range, above = trend
EMA_TREND_PERIOD = 50
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.20  # 20% position size
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def calculate_rsi(close, period):
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def calculate_atr(high, low, close, period):
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_adx(high, low, close, period):
    plus_dm = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    minus_dm = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/period, adjust=False, min_periods=period).mean() / \
              pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/period, adjust=False, min_periods=period).mean() / \
               pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    dx = (np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)) * 100
    adx = pd.Series(dx).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return adx

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # 4h EMA for trend direction
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=EMA_TREND_PERIOD, adjust=False, min_periods=EMA_TREND_PERIOD).mean().values
    trend_4h = np.where(close_4h > ema_4h, 1, -1)  # 1=uptrend, -1=downtrend
    trend_4h_aligned = align_htf_to_ltf(prices, df_4h, trend_4h)
    
    # 1d ADX for regime filter (range vs trend)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, ADX_PERIOD)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # LTF indicators (1h)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # RSI for mean reversion
    rsi = calculate_rsi(close, RSI_PERIOD)
    
    # Bollinger Bands
    sma = pd.Series(close).rolling(window=BB_PERIOD, min_periods=BB_PERIOD).mean().values
    std = pd.Series(close).rolling(window=BB_PERIOD, min_periods=BB_PERIOD).std().values
    bb_upper = sma + (BB_STD * std)
    bb_lower = sma - (BB_STD * std)
    
    # Volume confirmation
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR for risk management
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(RSI_PERIOD, BB_PERIOD, ADX_PERIOD, EMA_TREND_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(trend_4h_aligned[i]) or np.isnan(adx_1d_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Regime filter: only trade in ranging markets (ADX < 25)
        is_range = adx_1d_aligned[i] < ADX_THRESHOLD
        
        # Mean reversion signals only in range
        if is_range:
            # RSI extremes + Bollinger Band touches
            rsi_oversold = rsi[i] < RSI_OVERSOLD
            rsi_overbought = rsi[i] > RSI_OVERBOUGHT
            bb_lower_touch = close[i] <= bb_lower[i]
            bb_upper_touch = close[i] >= bb_upper[i]
            
            # Volume confirmation
            vol_confirmed = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
            
            # Long: oversold + lower BB touch + volume + aligned with 4h trend (avoid counter-trend)
            long_signal = rsi_oversold and bb_lower_touch and vol_confirmed and (trend_4h_aligned[i] == 1 or not vol_confirmed)
            # Short: overbought + upper BB touch + volume + aligned with 4h trend
            short_signal = rsi_overbought and bb_upper_touch and vol_confirmed and (trend_4h_aligned[i] == -1 or not vol_confirmed)
            
            # If volume not confirming, still allow trade if strongly aligned with trend
            if not vol_confirmed:
                long_signal = rsi_oversold and bb_lower_touch and (trend_4h_aligned[i] == 1)
                short_signal = rsi_overbought and bb_upper_touch and (trend_4h_aligned[i] == -1)
        else:
            # In trending markets, only trade pullbacks to mean (opposite logic)
            # Long on pullback in uptrend
            long_signal = (close[i] < sma[i]) and (trend_4h_aligned[i] == 1) and (rsi[i] < 50)
            # Short on pullback in downtrend
            short_signal = (close[i] > sma[i]) and (trend_4h_aligned[i] == -1) and (rsi[i] > 50)
            vol_confirmed = False  # Not used in trend mode
        
        # Generate signals
        if position == 0:
            if long_signal:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif short_signal:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals