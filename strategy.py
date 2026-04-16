#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA trend with weekly RSI filter and volume confirmation
# Long when: price > KAMA(10,2,30) AND weekly RSI(14) > 50 AND volume > 1.5x 20-day average
# Short when: price < KAMA(10,2,30) AND weekly RSI(14) < 50 AND volume > 1.5x 20-day average
# Exit: opposite KAMA cross or ATR(14) stoploss (2*ATR from entry)
# Uses discrete position size 0.25. Designed to capture medium-term trends with momentum and volume confirmation.
# Works in both bull and bear markets by requiring trend alignment (KAMA direction) and momentum filter (weekly RSI).
# Target: 30-100 total trades over 4 years (7-25/year) to minimize fee drag while maintaining edge.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d Indicators: KAMA(10,2,30) for trend ===
    # Efficiency Ratio: ER = |close - close[10]| / sum(|close - close[1]|) over 10 periods
    change = np.abs(np.diff(close, n=10))  # |close[i] - close[i-10]|
    volatility = np.sum(np.abs(np.diff(close, n=1)), axis=0)  # sum of |close[i] - close[i-1]| over 10 periods
    # Avoid division by zero
    er = np.divide(change, volatility, out=np.zeros_like(change, dtype=float), where=volatility!=0)
    # Smoothing constants: sc = [ER * (fastest - slowest) + slowest]^2
    fastest = 2.0 / (2 + 1)   # 2/(2+1) = 0.666...
    slowest = 2.0 / (30 + 1)  # 2/(30+1) = 0.0645...
    sc = (er * (fastest - slowest) + slowest) ** 2
    # KAMA: kama[i] = kama[i-1] + sc * (price[i] - kama[i-1])
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i-1] * (close[i] - kama[i-1]) if i-1 < len(sc) else kama[i-1]
    kama = np.concatenate([np.full(9, np.nan), kama[9:]])  # align with lookback
    
    # === 1w Indicators: RSI(14) for momentum filter ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    # RSI calculation: RS = avg_gain / avg_loss over 14 periods
    delta = np.diff(close_1w)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    # Wilder's smoothing (alpha = 1/14)
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[13] = np.mean(gain[0:14])  # first average
    avg_loss[13] = np.mean(loss[0:14])
    for i in range(14, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain, dtype=float), where=avg_loss!=0)
    rsi_1w = 100 - (100 / (1 + rs))
    rsi_1w = np.concatenate([np.full(14, np.nan), rsi_1w])  # align with lookback
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    rsi_above_50 = rsi_1w_aligned > 50
    rsi_below_50 = rsi_1w_aligned < 50
    
    # === 1d Indicators: Volume Spike (volume > 1.5x 20-period average) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # === 1d ATR for stoploss ===
    tr1 = pd.Series(high - low).values
    tr2 = pd.Series(np.abs(high - np.roll(close, 1))).values
    tr3 = pd.Series(np.abs(low - np.roll(close, 1))).values
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    # Set first TR to high-low since no previous close
    tr_1d[0] = high[0] - low[0]
    atr_1d_raw = pd.Series(tr_1d).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 50 periods needed)
    warmup = 100
    
    # Track position state and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(kama[i]) or np.isnan(rsi_1w_aligned[i]) or np.isnan(volume_spike[i]) or
            np.isnan(atr_1d_raw[i]) or not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        vol_spike = volume_spike[i]
        atr_val = atr_1d_raw[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price crosses below KAMA
            if price < kama[i]:
                exit_signal = True
            # ATR-based stoploss: 2*ATR below entry
            elif price < entry_price - 2.0 * atr_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price crosses above KAMA
            if price > kama[i]:
                exit_signal = True
            # ATR-based stoploss: 2*ATR above entry
            elif price > entry_price + 2.0 * atr_val:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price above KAMA AND weekly RSI > 50 AND volume spike
            if price > kama[i] and rsi_above_50[i] and vol_spike:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: Price below KAMA AND weekly RSI < 50 AND volume spike
            elif price < kama[i] and rsi_below_50[i] and vol_spike:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "1d_KAMA10_2_30_1wRSI14_VolumeSpike_V1"
timeframe = "1d"
leverage = 1.0