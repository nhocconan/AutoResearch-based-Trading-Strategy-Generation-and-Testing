#!/usr/bin/env python3
"""
Experiment #022: 4h ADX Regime-Adaptive + 1d HMA Trend Bias

Hypothesis: Markets alternate between trending (ADX>25) and ranging (ADX<20).
Use different logic per regime: trend-following breakouts when trending,
mean-reversion at channel bounds when ranging. 1d HMA provides directional bias.

Timeframe: 4h (REQUIRED for this experiment)
HTF: 1d via mtf_data helper (get_htf_data ONCE before loop)
Position sizing: 0.25 base, 0.30 max, discrete levels
Stoploss: 2.5*ATR trailing stop

Key innovation: ADX hysteresis (enter 25, exit 18) reduces whipsaw in transition zones.
Loose entry conditions ensure sufficient trade generation (avoiding 0-trade failure).
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_adx_regime_1d_hma_v1"
timeframe = "4h"
leverage = 1.0

def calculate_adx(high, low, close, period=14):
    """Calculate ADX using Wilder's smoothing method."""
    n = len(close)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        plus_move = high[i] - high[i - 1]
        minus_move = low[i - 1] - low[i]
        
        if plus_move > minus_move and plus_move > 0:
            plus_dm[i] = plus_move
        if minus_move > plus_move and minus_move > 0:
            minus_dm[i] = minus_move
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_di = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_di = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    plus_di = np.where(atr > 0, 100 * plus_di / atr, 0)
    minus_di = np.where(atr > 0, 100 * minus_di / atr, 0)
    
    dx = np.where((plus_di + minus_di) > 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (highest high, lowest low over period)."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    upper = high_s.rolling(window=period, min_periods=period).max().values
    lower = low_s.rolling(window=period, min_periods=period).min().values
    return upper, lower

def calculate_rsi(close, period=14):
    """Calculate RSI using standard formula."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50.0).values
    return rsi

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_hma(close, period=21):
    """Hull Moving Average for smoother trend."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (CRITICAL - Rule 1)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HMA for trend bias
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 4h indicators
    adx = calculate_adx(high, low, close, 14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    rsi = calculate_rsi(close, 14)
    atr = calculate_atr(high, low, close, 14)
    
    # Price position in Donchian channel (0=low, 1=high)
    donchian_range = donchian_upper - donchian_lower
    donchian_position = np.divide(close - donchian_lower, donchian_range, 
                                   out=np.zeros_like(close), where=donchian_range != 0)
    donchian_position = np.clip(donchian_position, 0, 1)
    
    signals = np.zeros(n)
    
    SIZE_BASE = 0.25
    SIZE_MAX = 0.30
    
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    highest_close = 0.0
    lowest_close = 0.0
    
    # ADX hysteresis tracking
    in_trend_regime = False
    in_range_regime = False
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(adx[i]) or np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(donchian_position[i]):
            signals[i] = 0.0
            continue
        
        # Regime detection with hysteresis
        if adx[i] > 25:
            in_trend_regime = True
            in_range_regime = False
        elif adx[i] < 18:
            in_range_regime = True
            in_trend_regime = False
        
        # Daily trend bias
        bull_trend = close[i] > hma_1d_aligned[i]
        bear_trend = close[i] < hma_1d_aligned[i]
        
        # Donchian breakout signals (slightly loose for more trades)
        breakout_long = close[i] >= donchian_upper[i] * 0.995
        breakout_short = close[i] <= donchian_lower[i] * 1.005
        
        # RSI extremes for mean reversion (loose thresholds)
        rsi_oversold = rsi[i] < 40
        rsi_overbought = rsi[i] > 60
        
        # Channel position extremes
        near_channel_low = donchian_position[i] < 0.15
        near_channel_high = donchian_position[i] > 0.85
        
        new_signal = 0.0
        
        # === TREND REGIME (ADX > 25) ===
        if in_trend_regime:
            # Long: Donchian breakout + daily bull trend
            if breakout_long and bull_trend:
                new_signal = SIZE_MAX
            # Short: Donchian breakout + daily bear trend
            elif breakout_short and bear_trend:
                new_signal = -SIZE_MAX
            # Weaker signals without daily confirmation (ensure trades happen)
            elif breakout_long:
                new_signal = SIZE_BASE
            elif breakout_short:
                new_signal = -SIZE_BASE
        
        # === RANGE REGIME (ADX < 18) ===
        elif in_range_regime:
            # Long: RSI oversold + near channel low
            if rsi_oversold and near_channel_low:
                new_signal = SIZE_BASE
            # Short: RSI overbought + near channel high
            elif rsi_overbought and near_channel_high:
                new_signal = -SIZE_BASE
            # Even looser: just RSI extremes
            elif rsi_oversold and bull_trend:
                new_signal = SIZE_BASE
            elif rsi_overbought and bear_trend:
                new_signal = -SIZE_BASE
        
        # === TRANSITION REGIME (18 <= ADX <= 25) ===
        # Keep existing position, don't open new ones
        else:
            new_signal = signals[i - 1] if i > 0 else 0.0
        
        # STOPLOSS: 2.5*ATR trailing
        if position_side > 0 and entry_price > 0:
            if close[i] > highest_close:
                highest_close = close[i]
            current_stop = highest_close - 2.5 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            if close[i] < trailing_stop:
                new_signal = 0.0
        
        if position_side < 0 and entry_price > 0:
            if lowest_close == 0.0 or close[i] < lowest_close:
                lowest_close = close[i]
            current_stop = lowest_close + 2.5 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            if close[i] > trailing_stop:
                new_signal = 0.0
        
        # Update position tracking
        prev_signal = signals[i - 1] if i > 0 else 0.0
        
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        elif new_signal == 0.0 and prev_signal != 0.0:
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
            highest_close = 0.0
            lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals