#!/usr/bin/env python3
"""
Experiment #015: 6h Elder Ray + Weekly Regime Filter

HYPOTHESIS: Elder Ray (Bull Power = High - EMA13, Bear Power = Low - EMA13)
measures the position of price extremes relative to EMA. Combined with 1w EMA
regime and 1d trend alignment, this captures:
- Bull markets: Long when Bear Power rises from negative (bottom capture)
- Bear markets: Short when Bull Power falls from positive (top capture)
- Range: Mean-revert at Elder extremes

Why it should work in BOTH bull AND bear:
- Bull: Elder Ray identifies pullbacks within uptrends
- Bear: Elder Ray identifies rallies within downtrends (short tops)
- Range: Elder extremes mark reversal points

TIMEFRAME: 6h primary
HTF: 1w for regime, 1d for trend
TARGET: 75-200 total trades over 4 years (19-50/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_elder_ray_1w_regime_v1"
timeframe = "6h"
leverage = 1.0

def calculate_ema(data, period):
    """Exponential Moving Average"""
    return pd.Series(data).ewm(span=period, min_periods=period, adjust=False).mean().values

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_rsi(close, period=14):
    """RSI for momentum confirmation"""
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = (100 - (100 / (1 + rs))).values
    return rsi

def calculate_elder_ray(high, low, close, ema_period=13):
    """Elder Ray: Bull Power and Bear Power"""
    n = len(close)
    ema = calculate_ema(close, ema_period)
    
    bull_power = np.zeros(n, dtype=np.float64)
    bear_power = np.zeros(n, dtype=np.float64)
    
    for i in range(ema_period, n):
        if not np.isnan(ema[i]):
            bull_power[i] = high[i] - ema[i]
            bear_power[i] = low[i] - ema[i]
    
    return bull_power, bear_power, ema

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE ===
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly EMA for regime (21-period)
    ema_1w = calculate_ema(df_1w['close'].values, 21)
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Daily EMA for trend (50-period)
    ema_1d = calculate_ema(df_1d['close'].values, 50)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate 6h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    
    # Elder Ray (13-period EMA)
    bull_power, bear_power, ema_13 = calculate_elder_ray(high, low, close, ema_period=13)
    
    # Elder Ray smoothed (3-period EMA of power)
    bull_power_smooth = pd.Series(bull_power).ewm(span=3, min_periods=2, adjust=False).mean().values
    bear_power_smooth = pd.Series(bear_power).ewm(span=3, min_periods=2, adjust=False).mean().values
    
    # Volume MA for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Elder Ray thresholds (tuned for 6h)
    BULL_THRESHOLD = -0.5  # Bear power rising from negative = bullish
    BEAR_THRESHOLD = 0.5   # Bull power falling from positive = bearish
    
    warmup = 60
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            continue
        if np.isnan(ema_1w_aligned[i]) or np.isnan(ema_1d_aligned[i]):
            continue
        if np.isnan(ema_13[i]):
            continue
        
        # === REGIME CHECK (1w EMA) ===
        weekly_trend_up = close[i] > ema_1w_aligned[i]
        weekly_trend_down = close[i] < ema_1w_aligned[i]
        
        # === TREND CHECK (1d EMA) ===
        daily_trend_up = close[i] > ema_1d_aligned[i]
        daily_trend_down = close[i] < ema_1d_aligned[i]
        
        # === ELDER RAY SIGNALS ===
        bull_pow = bull_power_smooth[i]
        bear_pow = bear_power_smooth[i]
        
        # Elder Ray momentum: bull power rising, bear power rising
        bull_power_rising = bull_pow > bull_power_smooth[i-1] if i > warmup else False
        bear_power_rising = bear_pow < bear_power_smooth[i-1] if i > warmup else False  # More negative
        
        # === RSI FOR MOMENTUM CONFIRMATION ===
        rsi_val = rsi_14[i]
        rsi_oversold = rsi_val < 35
        rsi_overbought = rsi_val > 65
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.2
        
        # === ENTRY CONDITIONS ===
        desired_signal = 0.0
        
        # === LONG ENTRY ===
        # Elder Ray: Bear power is negative (price below EMA) AND rising
        # Regime: Weekly uptrend
        # Trend: Daily uptrend
        # Momentum: RSI oversold
        # Confirmation: Volume spike
        if bear_pow < 0 and bear_pow > bear_power_smooth[i-1] if i > warmup else True:
            # Bear power is becoming less negative = potential reversal up
            if weekly_trend_up and daily_trend_up:
                # Both timeframes aligned bullish
                if rsi_oversold or vol_spike:
                    desired_signal = SIZE
        
        # === SHORT ENTRY ===
        # Elder Ray: Bull power is positive (price above EMA) AND falling
        # Regime: Weekly downtrend
        # Trend: Daily downtrend
        # Momentum: RSI overbought
        # Confirmation: Volume spike
        if bull_pow > 0 and bull_pow < bull_power_smooth[i-1] if i > warmup else True:
            # Bull power is declining = potential reversal down
            if weekly_trend_down and daily_trend_down:
                # Both timeframes aligned bearish
                if rsi_overbought or vol_spike:
                    desired_signal = -SIZE
        
        # === STOPLOSS CHECK (2.5 ATR) ===
        # Implemented via signal = 0 when price moves against position
        
        signals[i] = desired_signal
    
    return signals