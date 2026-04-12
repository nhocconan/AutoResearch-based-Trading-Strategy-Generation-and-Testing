# 4h_1d_ema50_rsi_mean_reversion_v1
# Strategy: Mean reversion on 4h timeframe using 1d EMA50 trend filter and RSI extremes
# Works in both bull and bear markets by trading reversions to trend with volatility filter
# Entry: RSI oversold/overbought + price relative to 1d EMA50 + volatility filter
# Exit: Trend change or volatility drop
# Position sizing: 0.25 long/short to manage drawdown
# Target: 20-50 trades/year to avoid fee drag

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Precompute hour filter for 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 1d ATR(14) for volatility filter
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = np.full(len(df_1d), np.nan)
    for i in range(14, len(df_1d)):
        atr_1d[i] = np.mean(tr[i-14:i+1])
    
    # Calculate 1d RSI(14) for mean reversion signals
    delta = np.diff(close_1d, prepend=np.nan)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.full(len(delta), np.nan)
    avg_loss = np.full(len(delta), np.nan)
    for i in range(14, len(delta)):
        if i == 14:
            avg_gain[i] = np.mean(gain[1:15])
            avg_loss[i] = np.mean(loss[1:15])
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_1d = 100 - (100 / (1 + rs))
    
    # Align 1d indicators to 4h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Precompute ATR MA for volatility filter
    atr_ma_20_1d = np.full(len(df_1d), np.nan)
    for j in range(19, len(df_1d)):
        start_idx = max(0, j-19)
        end_idx = j+1
        valid_slice = atr_1d[start_idx:end_idx]
        valid_vals = valid_slice[~np.isnan(valid_slice)]
        if len(valid_vals) >= 5:  # Require at least 5 valid values
            atr_ma_20_1d[j] = np.mean(valid_vals)
    atr_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(atr_1d_aligned[i]) or 
            np.isnan(rsi_1d_aligned[i]) or np.isnan(atr_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: 1d ATR > 0.3 * its 20-period MA (avoid low volatility)
        vol_filter = atr_1d_aligned[i] > 0.3 * atr_ma_20_1d_aligned[i]
        
        # Trend filter: price above/below 1d EMA50
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Mean reversion signals: RSI extremes
        rsi_oversold = rsi_1d_aligned[i] < 30
        rsi_overbought = rsi_1d_aligned[i] > 70
        
        # Entry conditions: 
        # Long: uptrend + volatility filter + RSI oversold
        # Short: downtrend + volatility filter + RSI overbought
        long_entry = uptrend and vol_filter and rsi_oversold
        short_entry = downtrend and vol_filter and rsi_overbought
        
        # Exit conditions: opposite trend or volatility drop
        long_exit = (not uptrend) or (not vol_filter)
        short_exit = (not downtrend) or (not vol_filter)
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_1d_ema50_rsi_mean_reversion_v1"
timeframe = "4h"
leverage = 1.0