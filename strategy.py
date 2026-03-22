#!/usr/bin/env python3
"""
Experiment #021: 1h Multi-Regime Strategy with 4h HMA Trend Filter
Hypothesis: Combine volatility spike mean-reversion with trend-following pullbacks.
Key insight: Previous 19 experiments failed due to overly strict filters (0 trades) or wrong regime logic.
This strategy uses LOOSE entry conditions to ensure 10+ trades per symbol:
  - 4h HMA for primary trend bias (proven in baseline)
  - Vol spike detection: ATR(7)/ATR(30) > 1.8 for mean-reversion entries
  - Trend pullback: price near EMA21 + RSI 30-60 (long) or 40-70 (short)
  - Breakout: Donchian(20) break with volume confirmation
  - Any 2 of 3 signals = entry (ensemble voting for more trades)
Position sizing: 0.25 discrete, stoploss at 2.5*ATR trailing.
Timeframe: 1h (REQUIRED), HTF: 4h via mtf_data helper.
Why this might work: Looser conditions ensure trades, ensemble reduces false signals,
4h HMA provides proven trend bias from baseline strategy.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_multi_regime_4h_hma_ensemble_v1"
timeframe = "1h"
leverage = 1.0

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    n = len(close)
    rsi = np.zeros(n)
    rsi[:] = np.nan
    
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    
    gains = np.where(delta > 0, delta, 0)
    losses = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gains).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(losses).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    mask = avg_loss > 0
    rs = np.zeros(n)
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rsi[mask] = 100 - (100 / (1 + rs[mask]))
    rsi[~mask] = 100.0
    
    return rsi

def calculate_ema(close, period):
    """Calculate EMA."""
    return pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return upper, lower, sma

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def calculate_volume_ma(volume, period=20):
    """Calculate volume moving average."""
    return pd.Series(volume).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1h indicators
    atr = calculate_atr(high, low, close, 14)
    atr_7 = calculate_atr(high, low, close, 7)
    atr_30 = calculate_atr(high, low, close, 30)
    rsi = calculate_rsi(close, 14)
    ema_21 = calculate_ema(close, 21)
    ema_50 = calculate_ema(close, 50)
    ema_200 = calculate_ema(close, 200)
    bb_upper, bb_lower, bb_mid = calculate_bollinger(close, 20, 2.0)
    donch_upper, donch_lower = calculate_donchian(high, low, 20)
    vol_ma = calculate_volume_ma(volume, 20)
    
    # Volatility ratio for spike detection
    vol_ratio = atr_7 / (atr_30 + 1e-10)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25
    SIZE_HALF = 0.125
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(ema_21[i]):
            signals[i] = 0.0
            continue
        
        # 4h trend bias (HTF) - main regime filter
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # 1h trend confirmation
        bull_trend_1h = close[i] > ema_50[i] and ema_21[i] > ema_50[i]
        bear_trend_1h = close[i] < ema_50[i] and ema_21[i] < ema_50[i]
        
        # Long-term trend filter
        above_200 = not np.isnan(ema_200[i]) and close[i] > ema_200[i]
        below_200 = not np.isnan(ema_200[i]) and close[i] < ema_200[i]
        
        # === SIGNAL 1: Volatility Spike Mean Reversion ===
        vol_spike = vol_ratio[i] > 1.8
        price_extreme_long = close[i] < bb_lower[i] * 1.005
        price_extreme_short = close[i] > bb_upper[i] * 0.995
        vol_meanrev_long = vol_spike and price_extreme_long
        vol_meanrev_short = vol_spike and price_extreme_short
        
        # === SIGNAL 2: Trend Pullback Entry ===
        price_near_ema21_long = close[i] <= ema_21[i] * 1.015 and close[i] >= ema_21[i] * 0.985
        price_near_ema21_short = close[i] >= ema_21[i] * 0.985 and close[i] <= ema_21[i] * 1.015
        rsi_pullback_long = 25 < rsi[i] < 65
        rsi_bounce_short = 35 < rsi[i] < 75
        trend_pullback_long = price_near_ema21_long and rsi_pullback_long
        trend_pullback_short = price_near_ema21_short and rsi_bounce_short
        
        # === SIGNAL 3: Donchian Breakout with Volume ===
        vol_above_avg = volume[i] > vol_ma[i] * 1.2 if not np.isnan(vol_ma[i]) else False
        breakout_long = close[i] > donch_upper[i-1] * 0.998 if not np.isnan(donch_upper[i-1]) else False
        breakout_short = close[i] < donch_lower[i-1] * 1.002 if not np.isnan(donch_lower[i-1]) else False
        breakout_long_conf = breakout_long and vol_above_avg
        breakout_short_conf = breakout_short and vol_above_avg
        
        # === ENSEMBLE VOTING (need 2 of 3 signals + trend alignment) ===
        long_signals = 0
        short_signals = 0
        
        if bull_trend_4h or bull_trend_1h:
            if vol_meanrev_long:
                long_signals += 1
            if trend_pullback_long:
                long_signals += 1
            if breakout_long_conf and above_200:
                long_signals += 1
        
        if bear_trend_4h or bear_trend_1h:
            if vol_meanrev_short:
                short_signals += 1
            if trend_pullback_short:
                short_signals += 1
            if breakout_short_conf and below_200:
                short_signals += 1
        
        # Also allow counter-trend mean reversion on extreme vol spikes
        if vol_ratio[i] > 2.5:
            if price_extreme_long and rsi[i] < 30:
                long_signals += 1
            if price_extreme_short and rsi[i] > 70:
                short_signals += 1
        
        new_signal = 0.0
        
        # Entry logic: need 2+ signals OR 1 extreme signal
        if long_signals >= 2 or (long_signals >= 1 and vol_ratio[i] > 2.5):
            new_signal = SIZE_BASE
        elif short_signals >= 2 or (short_signals >= 1 and vol_ratio[i] > 2.5):
            new_signal = -SIZE_BASE
        elif long_signals >= 1 and bull_trend_4h:
            new_signal = SIZE_HALF
        elif short_signals >= 1 and bear_trend_4h:
            new_signal = -SIZE_HALF
        
        # === STOPLOSS LOGIC (Rule 6) ===
        # Long position stoploss
        if position_side > 0 and entry_price > 0:
            if close[i] > highest_close:
                highest_close = close[i]
            
            current_stop = highest_close - 2.5 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            if close[i] < trailing_stop:
                new_signal = 0.0
        
        # Short position stoploss
        if position_side < 0 and entry_price > 0:
            if close[i] < lowest_close or lowest_close == 0.0:
                lowest_close = close[i]
            
            current_stop = lowest_close + 2.5 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            if close[i] > trailing_stop:
                new_signal = 0.0
        
        # Update position tracking AFTER signal calculation
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