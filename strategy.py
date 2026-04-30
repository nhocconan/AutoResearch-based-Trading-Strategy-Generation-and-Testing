#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h RSI(2) mean reversion with 4h trend filter and volume confirmation
# Uses discrete sizing 0.20 to minimize fee drag. Target: 60-150 total trades over 4 years (15-37/year).
# RSI(2) identifies extreme short-term reversals. 4h EMA50 filters counter-trend trades.
# Volume spike ensures institutional participation. Session filter (08-20 UTC) reduces noise.
# Works in both bull and bear via 4h trend filter - only long in uptrend, short in downtrend.

name = "1h_RSI2_4hEMA50_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate RSI(2) on 1h timeframe
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/2, adjust=False, min_periods=2).mean()
    avg_loss = loss.ewm(alpha=1/2, adjust=False, min_periods=2).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_2 = 100 - (100 / (1 + rs))
    rsi_2 = rsi_2.fillna(50).values  # Neutral when undefined
    
    # Calculate 4h EMA(50) for trend filter (HTF)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    ema_50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Volume confirmation: volume > 1.5x 20-period average (balanced frequency)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    # ATR for stoploss (14-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(20, 50, 14)  # warmup
    
    for i in range(start_idx, n):
        # Skip if indicators not ready or outside session
        if (np.isnan(rsi_2[i]) or np.isnan(ema_50_4h_aligned[i]) or
            np.isnan(vol_ma_20[i]) or np.isnan(atr_14[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_rsi = rsi_2[i]
        curr_ema_50_4h = ema_50_4h_aligned[i]
        curr_volume_spike = volume_spike[i]
        curr_atr = atr_14[i]
        
        if position == 0:  # Flat - look for new entries
            # Only trade on volume spike with RSI extreme and 4h EMA50 trend filter
            if curr_volume_spike:
                # Bullish: RSI < 10 (oversold) + close above 4h EMA50 (uptrend)
                if curr_rsi < 10 and curr_close > curr_ema_50_4h:
                    signals[i] = 0.20
                    position = 1
                    entry_price = curr_close
                # Bearish: RSI > 90 (overbought) + close below 4h EMA50 (downtrend)
                elif curr_rsi > 90 and curr_close < curr_ema_50_4h:
                    signals[i] = -0.20
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Stoploss: 2 * ATR below entry
            stop_loss = entry_price - 2.0 * curr_atr
            # Exit: Stoploss hit OR RSI returns to neutral (50) OR loses 4h trend
            if curr_low <= stop_loss or curr_rsi >= 50 or curr_close < curr_ema_50_4h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Stoploss: 2 * ATR above entry
            stop_loss = entry_price + 2.0 * curr_atr
            # Exit: Stoploss hit OR RSI returns to neutral (50) OR loses 4h trend
            if curr_high >= stop_loss or curr_rsi <= 50 or curr_close > curr_ema_50_4h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals