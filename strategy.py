#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_14014_1h_rsi_volume_filter_v1"
timeframe = "1h"
leverage = 1.0

def calculate_rsi(close, period):
    """Calculate RSI with Wilder's smoothing"""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 4h data for trend filter (once before loop)
    df_4h = get_htf_data(prices, '4h')
    # Calculate 4h EMA(50) for trend bias
    ema_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Load 1d data for additional trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    # Calculate 1d EMA(50) for trend bias
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate indicators on 1h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # RSI(14) for momentum
    rsi = calculate_rsi(close, 14)
    
    # ATR(14) for stop loss
    atr = calculate_atr(high, low, close, 14)
    
    # Volume confirmation (20-period average)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC (pre-compute hours)
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period (max of all lookbacks)
    start = max(50, 14, 20) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(ema_4h_aligned[i]) or np.isnan(ema_1d_aligned[i]) or np.isnan(rsi[i]) or \
           np.isnan(volume_ma[i]) or np.isnan(atr[i]):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Check stops
        if position == 1:  # long position
            # Check stop loss
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # short position
            # Check stop loss
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        # Determine trend bias from 4h and 1d EMA (both must agree)
        bullish_trend = close[i] > ema_4h_aligned[i] and close[i] > ema_1d_aligned[i]
        bearish_trend = close[i] < ema_4h_aligned[i] and close[i] < ema_1d_aligned[i]
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * 1.5)
        
        # RSI conditions (avoid extremes, look for momentum)
        rsi_not_overbought = rsi[i] < 70
        rsi_not_oversold = rsi[i] > 30
        
        # Entry signals with trend, volume, and RSI filters
        long_signal = bullish_trend and volume_ok and rsi_not_oversold and rsi[i] > 50
        short_signal = bearish_trend and volume_ok and rsi_not_overbought and rsi[i] < 50
        
        # Generate signals
        if position == 0:
            if long_signal and in_session:
                signals[i] = 0.20
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (2.0 * atr[i])
            elif short_signal and in_session:
                signals[i] = -0.20
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (2.0 * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long on trend change or RSI overbought
            if not bullish_trend or rsi[i] >= 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short on trend change or RSI oversold
            if not bearish_trend or rsi[i] <= 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

</think>

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_14014_1h_rsi_volume_filter_v1"
timeframe = "1h"
leverage = 1.0

def calculate_rsi(close, period):
    """Calculate RSI with Wilder's smoothing"""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 4h data for trend filter (once before loop)
    df_4h = get_htf_data(prices, '4h')
    # Calculate 4h EMA(50) for trend bias
    ema_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Load 1d data for additional trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    # Calculate 1d EMA(50) for trend bias
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate indicators on 1h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # RSI(14) for momentum
    rsi = calculate_rsi(close, 14)
    
    # ATR(14) for stop loss
    atr = calculate_atr(high, low, close, 14)
    
    # Volume confirmation (20-period average)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC (pre-compute hours)
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period (max of all lookbacks)
    start = max(50, 14, 20) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(ema_4h_aligned[i]) or np.isnan(ema_1d_aligned[i]) or np.isnan(rsi[i]) or \
           np.isnan(volume_ma[i]) or np.isnan(atr[i]):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Check stops
        if position == 1:  # long position
            # Check stop loss
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # short position
            # Check stop loss
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        # Determine trend bias from 4h and 1d EMA (both must agree)
        bullish_trend = close[i] > ema_4h_aligned[i] and close[i] > ema_1d_aligned[i]
        bearish_trend = close[i] < ema_4h_aligned[i] and close[i] < ema_1d_aligned[i]
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * 1.5)
        
        # RSI conditions (avoid extremes, look for momentum)
        rsi_not_overbought = rsi[i] < 70
        rsi_not_oversold = rsi[i] > 30
        
        # Entry signals with trend, volume, and RSI filters
        long_signal = bullish_trend and volume_ok and rsi_not_oversold and rsi[i] > 50
        short_signal = bearish_trend and volume_ok and rsi_not_overbought and rsi[i] < 50
        
        # Generate signals
        if position == 0:
            if long_signal and in_session:
                signals[i] = 0.20
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (2.0 * atr[i])
            elif short_signal and in_session:
                signals[i] = -0.20
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (2.0 * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long on trend change or RSI overbought
            if not bullish_trend or rsi[i] >= 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short on trend change or RSI oversold
            if not bearish_trend or rsi[i] <= 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

</think>