#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h volume-weighted RSI (VW-RSI) with 1d trend filter and volatility filter.
# VW-RSI uses volume-weighted average price (VWAP) instead of close for RSI calculation.
# In trending markets, volume confirms strength; in ranging markets, extremes signal reversals.
# 1d EMA(50) slope determines trend direction: only take VW-RSI signals in trend direction.
# Volatility filter (ATR ratio) avoids low-volatility chop.
# Works in bull markets (buy strength on dips) and bear markets (sell weakness on rallies).

name = "exp_13611_6h_vwrsi_1d_trend_volfilt_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
VW_RSI_PERIOD = 14
TREND_EMA_PERIOD = 50
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
ATR_PERIOD = 14
ATR_RATIO_PERIOD = 30
ATR_RATIO_THRESHOLD = 0.5
SIGNAL_SIZE = 0.25
ATR_STOP_MULTIPLIER = 2.0

def calculate_vw_rsi(high, low, close, volume, period):
    """Calculate Volume-Weighted RSI using VWAP-like price"""
    # Calculate typical price
    typical_price = (high + low + close) / 3.0
    # Volume-weighted price (approximation of VWAP)
    vw_price = np.zeros_like(typical_price)
    cumulative_volume = np.zeros_like(volume)
    cumulative_vwp = np.zeros_like(typical_price)
    
    # Calculate cumulative values for VWAP-like calculation
    for i in range(len(typical_price)):
        cumulative_volume[i] = volume[i] + (cumulative_volume[i-1] if i > 0 else 0)
        cumulative_vwp[i] = typical_price[i] * volume[i] + (cumulative_vwp[i-1] if i > 0 else 0)
        if cumulative_volume[i] > 0:
            vw_price[i] = cumulative_vwp[i] / cumulative_volume[i]
        else:
            vw_price[i] = typical_price[i]
    
    # Calculate RSI on volume-weighted price
    delta = np.diff(vw_price, prepend=vw_price[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = pd.Series(gain).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_ema(close, period):
    """Calculate EMA"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for trend filter ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA for trend filter
    close_1d = df_1d['close'].values
    ema_1d = calculate_ema(close_1d, TREND_EMA_PERIOD)
    ema_1d_slope = np.diff(ema_1d, prepend=ema_1d[0])  # slope approximation
    ema_1d_slope_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_slope)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume-weighted RSI
    vw_rsi = calculate_vw_rsi(high, low, close, volume, VW_RSI_PERIOD)
    
    # ATR for stop loss and volatility filter
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    atr_long = calculate_atr(high, low, close, ATR_RATIO_PERIOD)
    atr_ratio = atr / (atr_long + 1e-10)  # short-term ATR / long-term ATR
    
    # Volume MA
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(VW_RSI_PERIOD, TREND_EMA_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD, ATR_RATIO_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(ema_1d_slope_aligned[i]) or np.isnan(vw_rsi[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(atr_ratio[i])):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
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
        
        # Filters
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD)
        volatility_ok = atr_ratio[i] > ATR_RATIO_THRESHOLD  # avoid low volatility
        
        # Trend direction from 1d EMA slope
        uptrend = ema_1d_slope_aligned[i] > 0
        downtrend = ema_1d_slope_aligned[i] < 0
        
        # VW-RSI signals: oversold/overbought with trend
        # In uptrend: buy when VW-RSI crosses above 30 from below (end of pullback)
        # In downtrend: sell when VW-RSI crosses below 70 from above (end of bounce)
        if i > 0 and not np.isnan(vw_rsi[i-1]):
            vw_rsi_prev = vw_rsi[i-1]
            vw_rsi_curr = vw_rsi[i]
            
            # Long signal: VW-RSI crosses above 30 in uptrend
            long_signal = volume_ok and volatility_ok and uptrend and vw_rsi_prev <= 30 and vw_rsi_curr > 30
            
            # Short signal: VW-RSI crosses below 70 in downtrend
            short_signal = volume_ok and volatility_ok and downtrend and vw_rsi_prev >= 70 and vw_rsi_curr < 70
        else:
            long_signal = False
            short_signal = False
        
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
            # Exit long on opposite VW-RSI signal or stop loss
            if i > 0 and not np.isnan(vw_rsi[i-1]):
                vw_rsi_prev = vw_rsi[i-1]
                vw_rsi_curr = vw_rsi[i]
                # Exit if VW-RSI crosses below 70 (overbought in uptrend)
                if vw_rsi_prev < 70 and vw_rsi_curr >= 70:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = SIGNAL_SIZE
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            # Exit short on opposite VW-RSI signal or stop loss
            if i > 0 and not np.isnan(vw_rsi[i-1]):
                vw_rsi_prev = vw_rsi[i-1]
                vw_rsi_curr = vw_rsi[i]
                # Exit if VW-RSI crosses above 30 (oversold in downtrend)
                if vw_rsi_prev > 30 and vw_rsi_curr <= 30:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -SIGNAL_SIZE
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals