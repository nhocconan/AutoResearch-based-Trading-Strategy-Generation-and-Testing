#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h RSI(2) extreme reversals with 4h trend filter and volume confirmation.
# RSI(2) captures short-term overbought/oversold conditions. In strong trends, extreme RSI(2) values
# often precede short-term reversals. Use 4h EMA(50) slope for trend direction: only take RSI(2) 
# reversals in trend direction. In uptrend: buy when RSI(2) crosses above 10 from below (oversold bounce).
# In downtrend: sell when RSI(2) crosses below 90 from above (overbought rejection).
# Volume confirmation ensures institutional participation. Session filter (08-20 UTC) reduces noise.
# Works in bull markets (buy dips in uptrend) and bear markets (sell rallies in downtrend).
# Target: 15-37 trades/year by using strict RSI(2) extremes (<10 or >90) + trend + volume + session.

name = "exp_13614_1h_rsi2_4h_trend_vol_sess_v1"
timeframe = "1h"
leverage = 1.0

# Parameters
RSI_PERIOD = 2
TREND_EMA_PERIOD = 50
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
RSI_OVERBOUGHT = 90
RSI_OVERSOLD = 10
SIGNAL_SIZE = 0.20
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_rsi(close, period):
    """Calculate RSI"""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def calculate_ema(close, period):
    """Calculate EMA"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]  # First TR is just high-low
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 4h data for trend filter ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h EMA for trend filter
    close_4h = df_4h['close'].values
    ema_4h = calculate_ema(close_4h, TREND_EMA_PERIOD)
    ema_4h_slope = np.diff(ema_4h, prepend=ema_4h[0])  # slope approximation
    ema_4h_slope_aligned = align_htf_to_ltf(prices, df_4h, ema_4h_slope)
    
    # Calculate 1h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # RSI(2)
    rsi = calculate_rsi(close, RSI_PERIOD)
    
    # ATR for stop loss
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Volume MA
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(RSI_PERIOD, TREND_EMA_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(ema_4h_slope_aligned[i]) or np.isnan(rsi[i]) or np.isnan(volume_ma[i]):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
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
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD)
        
        # Trend direction from 4h EMA slope
        uptrend = ema_4h_slope_aligned[i] > 0
        downtrend = ema_4h_slope_aligned[i] < 0
        
        # RSI(2) signals: extreme reversals
        # Avoid lookback by checking current and previous values
        if i > 0 and not np.isnan(rsi[i-1]):
            rsi_prev = rsi[i-1]
            rsi_curr = rsi[i]
            
            # Long signal: RSI(2) crosses above 10 from below in uptrend
            long_signal = volume_ok and in_session and uptrend and rsi_prev <= RSI_OVERSOLD and rsi_curr > RSI_OVERSOLD
            
            # Short signal: RSI(2) crosses below 90 from above in downtrend
            short_signal = volume_ok and in_session and downtrend and rsi_prev >= RSI_OVERBOUGHT and rsi_curr < RSI_OVERBOUGHT
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
            # Exit long on opposite RSI(2) signal or stop loss
            if i > 0 and not np.isnan(rsi[i-1]):
                rsi_prev = rsi[i-1]
                rsi_curr = rsi[i]
                # Exit if RSI(2) crosses below 90 (overbought - end of bounce)
                if rsi_prev < RSI_OVERBOUGHT and rsi_curr >= RSI_OVERBOUGHT:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = SIGNAL_SIZE
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            # Exit short on opposite RSI(2) signal or stop loss
            if i > 0 and not np.isnan(rsi[i-1]):
                rsi_prev = rsi[i-1]
                rsi_curr = rsi[i]
                # Exit if RSI(2) crosses above 10 (oversold - end of decline)
                if rsi_prev > RSI_OVERSOLD and rsi_curr <= RSI_OVERSOLD:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -SIGNAL_SIZE
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals