#!/usr/bin/env python3
"""
exp_7434_1h_rsi_divergence_volatility_filter_v1
Hypothesis: 1h RSI divergence (bullish/bearish) with volatility filter and volume confirmation.
Uses 4h EMA(50) for trend direction to avoid counter-trend trades, targeting 80-150 trades over 4 years.
Designed to work in both bull and bear markets by combining mean reversion (RSI) with trend filter.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7434_1h_rsi_divergence_volatility_filter_v1"
timeframe = "1h"
leverage = 1.0

# Parameters
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
VOLATILITY_LOOKBACK = 20
VOLATILITY_THRESHOLD = 1.5
VOLUME_CONFIRM = 1.5
EMA_TREND_PERIOD = 50
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0
SIGNAL_SIZE = 0.25
MAX_HOLD_BARS = 24  # 24 hours max hold

def calculate_rsi(prices, period=14):
    """Calculate RSI with proper handling"""
    delta = np.diff(prices, prepend=prices[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 4h for trend filter
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h EMA for trend direction
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=EMA_TREND_PERIOD, adjust=False, min_periods=EMA_TREND_PERIOD).mean().values
    
    # Align 4h EMA to 1h timeframe
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Calculate 1h indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # RSI
    rsi = calculate_rsi(close, RSI_PERIOD)
    
    # Volatility (ATR-based)
    tr1 = high - low
    tr2 = np.abs(np.roll(close, 1) - high)
    tr3 = np.abs(np.roll(close, 1) - low)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    # Volatility filter: current ATR vs average ATR
    atr_ma = pd.Series(atr).rolling(window=VOLATILITY_LOOKBACK, min_periods=VOLATILITY_LOOKBACK).mean().values
    volatility_filter = atr > (atr_ma * VOLATILITY_THRESHOLD) if not np.isnan(atr_ma[-1]) else False
    
    # Volume confirmation
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (volume_ma * VOLUME_CONFIRM) if not np.isnan(volume_ma[-1]) else False
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = max(RSI_PERIOD, ATR_PERIOD, VOLATILITY_LOOKBACK, 20) + 1
    
    for i in range(start, n):
        bars_since_entry += 1
        
        # Skip if HTF data not available
        if np.isnan(ema_4h_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= entry_price - ATR_STOP_MULTIPLIER * atr[i]:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
                continue
        elif position == -1:  # short position
            if close[i] >= entry_price + ATR_STOP_MULTIPLIER * atr[i]:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
                continue
                
        # Time-based exit
        if position != 0 and bars_since_entry >= MAX_HOLD_BARS:
            signals[i] = 0.0
            position = 0
            bars_since_entry = 0
            continue
            
        # Check volatility and volume filters
        vol_filter = volatility_filter[i] if hasattr(volatility_filter, '__len__') else volatility_filter
        vol_conf = volume_confirmed[i] if hasattr(volume_confirmed, '__len__') else volume_confirmed
        
        # RSI divergence detection (simplified: look for RSI extremes with price action)
        # Bullish divergence: price makes lower low, RSI makes higher low
        # Bearish divergence: price makes higher high, RSI makes lower high
        # Using 3-bar lookback for divergence
        
        if i >= 3:
            # Price action
            price_lower_low = low[i] < low[i-1] and low[i-1] < low[i-2]
            price_higher_high = high[i] > high[i-1] and high[i-1] > high[i-2]
            
            # RSI action
            rsi_higher_low = rsi[i] > rsi[i-1] and rsi[i-1] < rsi[i-2]
            rsi_lower_high = rsi[i] < rsi[i-1] and rsi[i-1] > rsi[i-2]
            
            # Bullish divergence: price down, RSI up
            bullish_div = price_lower_low and rsi_higher_low
            
            # Bearish divergence: price up, RSI down
            bearish_div = price_higher_high and rsi_lower_high
            
            # Additional filters
            oversold = rsi[i] < RSI_OVERSOLD
            overbought = rsi[i] > RSI_OVERBOUGHT
            
            # Trend filter from 4h EMA
            uptrend = close[i] > ema_4h_aligned[i]
            downtrend = close[i] < ema_4h_aligned[i]
            
            # Entry conditions
            if position == 0:
                # Long: bullish divergence in oversold area OR oversold bounce in uptrend
                long_signal = (bullish_div and oversold) or (oversold and uptrend and vol_filter and vol_conf)
                
                # Short: bearish divergence in overbought area OR overbought rejection in downtrend
                short_signal = (bearish_div and overbought) or (overbought and downtrend and vol_filter and vol_conf)
                
                if long_signal:
                    signals[i] = SIGNAL_SIZE
                    position = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                elif short_signal:
                    signals[i] = -SIGNAL_SIZE
                    position = -1
                    entry_price = close[i]
                    bars_since_entry = 0
                else:
                    signals[i] = 0.0
            else:
                # Hold current position
                signals[i] = position * SIGNAL_SIZE
        else:
            signals[i] = 0.0
    
    return signals