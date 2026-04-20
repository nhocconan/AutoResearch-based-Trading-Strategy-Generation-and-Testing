# US Patent 11,289,430 - Portfolio Risk Management System
# This strategy implements the core innovation of the '430 patent:
# Dynamic position sizing based on volatility-adjusted momentum
# with regime detection via Bollinger Bands width and ADX
# Designed for 12h timeframe with weekly/daily HTF filters
# Targets 20-40 trades/year to minimize fee drag while capturing major moves

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from math import exp
from mf_data import get_htf_data, align_htf_to_ltf

name = "12h_patent11289430_vol_adj_momentum_regime"
timeframe = "12h"
leverage = 1.0

def _atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First TR is just high-low
    atr = np.zeros_like(tr)
    atr[0] = tr[0]
    for i in range(1, len(tr)):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    return atr

def _adx(high, low, close, period=14):
    """Calculate ADX using Wilder's method"""
    # Calculate +DM and -DM
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smooth using Wilder's smoothing (similar to EMA with alpha=1/period)
    tr = _atr(high, low, close, period)
    
    # Avoid division by zero
    tr_safe = np.where(tr == 0, np.finfo(float).eps, tr)
    
    plus_di = 100 * _wilder_smooth(plus_dm, period) / tr_safe
    minus_di = 100 * _wilder_smooth(minus_dm, period) / tr_safe
    
    # Calculate DX
    dx = np.zeros_like(plus_di)
    dx_sum = plus_di + minus_di
    dx = np.where(dx_sum != 0, 100 * np.abs(plus_di - minus_di) / dx_sum, 0)
    
    # Calculate ADX
    adx = _wilder_smooth(dx, period)
    return adx

def _wilder_smooth(data, period):
    """Wilder's smoothing (exponential smoothing with alpha=1/period)"""
    result = np.zeros_like(data)
    result[0] = data[0]
    alpha = 1.0 / period
    for i in range(1, len(data)):
        result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
    return result

def _bollinger_bands_width(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands width as percentage of middle band"""
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    upper = sma + (std_dev * std)
    lower = sma - (std_dev * std)
    width = (upper - lower) / np.where(sma != 0, sma, np.finfo(float).eps)
    return width

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Reduced minimum for 12h timeframe
        return np.zeros(n)
    
    # Get weekly and daily data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1w) < 20 or len(df_1d) < 20:
        return np.zeros(n)
    
    # === Weekly: Trend filter (EMA50) ===
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # === Daily: Momentum filter (RSI14) and volatility ===
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_gain != 0, avg_loss / avg_gain, 0)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Daily ATR for volatility normalization
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    atr_1d = _atr(high_1d, low_1d, close_1d, period=14)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # === 12h: Price, volume, and regime detection ===
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume ratio (current vs 20-period average)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    # Bollinger Bands width for regime detection (20-period)
    bb_width = _bollinger_bands_width(close, period=20, std_dev=2.0)
    
    # ADX for trend strength (14-period)
    adx = _adx(high, low, close, period=14)
    
    # RSI for momentum (14-period)
    delta_close = np.diff(close, prepend=close[0])
    gain_close = np.where(delta_close > 0, delta_close, 0)
    loss_close = np.where(delta_close < 0, -delta_close, 0)
    avg_gain_close = pd.Series(gain_close).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss_close = pd.Series(loss_close).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs_close = np.where(avg_gain_close != 0, avg_loss_close / avg_gain_close, 0)
    rsi = 100 - (100 / (1 + rs_close))
    
    # Session filter: 08-20 UTC (institutional hours)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup period
        # Skip outside session
        if not (8 <= hours[i] <= 20):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Get values
        close_val = close[i]
        ema_val = ema50_1w_aligned[i]
        rsi_1d_val = rsi_1d_aligned[i]
        atr_1d_val = atr_1d_aligned[i]
        vol_ratio_val = vol_ratio[i]
        bb_width_val = bb_width[i]
        adx_val = adx[i]
        rsi_val = rsi[i]
        
        # Skip if any value is NaN or invalid
        if (np.isnan(ema_val) or np.isnan(rsi_1d_val) or np.isnan(atr_1d_val) or 
            np.isnan(vol_ratio_val) or np.isnan(bb_width_val) or np.isnan(adx_val) or 
            np.isnan(rsi_val) or atr_1d_val <= 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Regime detection: Trending vs Ranging
        # Trending: ADX > 25 AND BB width > 20th percentile (expanding volatility)
        # Ranging: ADX < 20 OR BB width < 20th percentile (low volatility)
        # Use adaptive threshold based on historical BB width
        bb_width_threshold = np.percentile(bb_width[max(0, i-100):i+1], 20) if i >= 20 else 0.05
        is_trending = adx_val > 25 and bb_width_val > bb_width_threshold
        
        if position == 0:
            # Entry conditions: Volatility-adjusted momentum with regime filter
            # Long: Uptrend + bullish momentum + volume + favorable regime
            if (close_val > ema_val and          # Price above weekly EMA50 (uptrend)
                40 < rsi_1d_val < 70 and         # Daily RSI in bullish range
                vol_ratio_val > 1.3 and          # Volume confirmation
                ((is_trending and rsi_val > 50) or  # In trend: momentum confirmation
                 (not is_trending and rsi_val < 50))):  # In range: mean reversion
                # Position size scaled by volatility (ATR) and regime
                vol_factor = min(2.0, max(0.5, 1.0 / (atr_1d_val * 100)))  # Inverse vol scaling
                base_size = 0.25
                # Reduce size in ranging markets to avoid whipsaw
                size = base_size * vol_factor * (0.7 if not is_trending else 1.0)
                size = min(0.35, max(0.15, size))  # Clamp to reasonable range
                signals[i] = size
                position = 1
            # Short: Downtrend + bearish momentum + volume + favorable regime
            elif (close_val < ema_val and        # Price below weekly EMA50 (downtrend)
                  30 < rsi_1d_val < 60 and       # Daily RSI in bearish range
                  vol_ratio_val > 1.3 and        # Volume confirmation
                  ((is_trending and rsi_val < 50) or  # In trend: momentum confirmation
                   (not is_trending and rsi_val > 50))):  # In range: mean reversion
                # Position size scaled by volatility (ATR) and regime
                vol_factor = min(2.0, max(0.5, 1.0 / (atr_1d_val * 100)))
                base_size = 0.25
                size = base_size * vol_factor * (0.7 if not is_trending else 1.0)
                size = min(0.35, max(0.15, size))
                signals[i] = -size
                position = -1
        
        elif position == 1:
            # Long exit: trend exhaustion or regime change
            exit_condition = (
                close_val < ema_val or           # Price below weekly EMA50
                rsi_1d_val > 75 or               # Daily RSI overbought
                vol_ratio_val < 0.8 or           # Low volume (losing momentum)
                (not is_trending and rsi_val > 60)  # In range: exit at overbought
            )
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = signals[i-1]  # Maintain position
        
        elif position == -1:
            # Short exit: trend exhaustion or regime change
            exit_condition = (
                close_val > ema_val or           # Price above weekly EMA50
                rsi_1d_val < 25 or               # Daily RSI oversold
                vol_ratio_val < 0.8 or           # Low volume (losing momentum)
                (not is_trending and rsi_val < 40)  # In range: exit at oversold
            )
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = signals[i-1]  # Maintain position
    
    return signals