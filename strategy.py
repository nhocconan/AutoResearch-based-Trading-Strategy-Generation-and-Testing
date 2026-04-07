#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-hour volume-weighted average price (VWAP) deviation with 4-hour trend filter and daily volume regime filter
# Long when price > VWAP + 0.5*ATR, 4h close > 4h EMA50 (uptrend), and daily volume > 20-day average volume
# Short when price < VWAP - 0.5*ATR, 4h close < 4h EMA50 (downtrend), and daily volume > 20-day average volume
# Exit when price crosses VWAP or trend changes
# Stoploss at 2.0 * ATR(14)
# Position size: 0.20 (20% of capital)
# Uses 4h EMA50 for trend filter, daily volume for regime filter, and 1h VWAP for entry timing
# Target: 60-150 total trades over 4 years (15-38/year)

name = "1h_vwap_dev_4h_trend_1d_vol_regime_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1h VWAP calculation
    typical_price = (high + low + close) / 3.0
    vp = typical_price * volume
    cum_vp = np.nancumsum(vp)
    cum_vol = np.nancumsum(volume)
    vwap = np.divide(cum_vp, cum_vol, out=np.zeros_like(cum_vp), where=cum_vol!=0)
    
    # 1h ATR(14) for volatility and stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # 1d data for volume regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    volume_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if required data not available
        if (np.isnan(vwap[i]) or np.isnan(atr[i]) or 
            np.isnan(ema50_4h_aligned[i]) or np.isnan(volume_ma_1d_aligned[i])):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2.0 * ATR
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price crosses below VWAP or trend changes
            elif close[i] < vwap[i] or close[i] < ema50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            # Stoploss: 2.0 * ATR
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price crosses above VWAP or trend changes
            elif close[i] > vwap[i] or close[i] > ema50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.20
        else:
            # Look for entries with VWAP deviation, trend alignment, and volume regime
            # Long: price > VWAP + 0.5*ATR, 4h uptrend, high volume regime
            if (close[i] > vwap[i] + 0.5 * atr[i] and
                close[i] > ema50_4h_aligned[i] and
                volume[i] > volume_ma_1d_aligned[i]):
                signals[i] = 0.20
                position = 1
                entry_price = close[i]
            # Short: price < VWAP - 0.5*ATR, 4h downtrend, high volume regime
            elif (close[i] < vwap[i] - 0.5 * atr[i] and
                  close[i] < ema50_4h_aligned[i] and
                  volume[i] > volume_ma_1d_aligned[i]):
                signals[i] = -0.20
                position = -1
                entry_price = close[i]
    
    return signals