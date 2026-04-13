#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h mean reversion strategy with 4h trend filter and 1d volatility regime filter.
# Uses RSI(14) on 1h for entry signals, 4h ADX(14) to filter trend strength (avoid whipsaw),
# and 1d ATR ratio (ATR7/ATR30) to detect high volatility regimes for mean reversion.
# In low volatility regimes (ATR7/ATR30 < 0.8), mean reversion works better.
# Entry: RSI < 30 for long, RSI > 70 for short, only when 4h ADX < 25 (weak trend) and 1d ATR ratio < 0.8.
# Exit: RSI crosses back to neutral (40-60 range) or stop loss via ATR.
# Position size: 0.20 (20%) to manage drawdown. Session filter: 08-20 UTC to avoid low liquidity.
# Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-calculate session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # 1h RSI(14)
    def calculate_rsi(prices, period=14):
        delta = np.diff(prices)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.zeros_like(prices)
        avg_loss = np.zeros_like(prices)
        avg_gain[period] = np.mean(gain[:period])
        avg_loss[period] = np.mean(loss[:period])
        for i in range(period + 1, len(prices)):
            avg_gain[i] = (avg_gain[i-1] * (period - 1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period - 1) + loss[i-1]) / period
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi = calculate_rsi(close, 14)
    
    # 4h data for ADX(14) trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # True Range and ATR for ADX calculation
    tr1 = np.abs(high_4h[1:] - low_4h[1:])
    tr2 = np.abs(high_4h[1:] - close_4h[:-1])
    tr3 = np.abs(low_4h[1:] - close_4h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index
    
    atr_period = 14
    atr = np.zeros_like(high_4h)
    atr[atr_period] = np.nanmean(tr[1:atr_period+1])  # skip first NaN
    for i in range(atr_period + 1, len(high_4h)):
        atr[i] = (atr[i-1] * (atr_period - 1) + tr[i]) / atr_period
    
    # Directional Movement
    up_move = high_4h[1:] - high_4h[:-1]
    down_move = low_4h[:-1] - low_4h[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    # Directional Indicators
    plus_di = 100 * np.where(atr != 0, pd.Series(plus_dm).ewm(span=atr_period, adjust=False).mean() / atr, 0)
    minus_di = 100 * np.where(atr != 0, pd.Series(minus_dm).ewm(span=atr_period, adjust=False).mean() / atr, 0)
    dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = pd.Series(dx).ewm(span=atr_period, adjust=False).mean().values
    
    # Align 4h ADX to 1h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_4h, adx)
    
    # 1d data for ATR ratio volatility regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ATR for 1d
    tr1_1d = np.abs(high_1d[1:] - low_1d[1:])
    tr2_1d = np.abs(high_1d[1:] - close_1d[:-1])
    tr3_1d = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))
    tr_1d = np.concatenate([[np.nan], tr_1d])
    
    atr7 = np.zeros_like(high_1d)
    atr30 = np.zeros_like(high_1d)
    atr7[7] = np.nanmean(tr_1d[1:8])
    atr30[30] = np.nanmean(tr_1d[1:31])
    for i in range(8, len(high_1d)):
        atr7[i] = (atr7[i-1] * 6 + tr_1d[i]) / 7
    for i in range(31, len(high_1d)):
        atr30[i] = (atr30[i-1] * 29 + tr_1d[i]) / 30
    
    # ATR ratio (ATR7/ATR30) - low when volatility is decreasing
    atr_ratio = np.where(atr30 != 0, atr7 / atr30, 1.0)
    
    # Align 1d ATR ratio to 1h timeframe
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.20  # 20% position size
    
    for i in range(30, n):  # start after warmup
        # Skip if not in trading session or data not ready
        if not in_session[i] or np.isnan(rsi[i]) or np.isnan(adx_aligned[i]) or np.isnan(atr_ratio_aligned[i]):
            signals[i] = 0.0
            continue
        
        rsi_val = rsi[i]
        adx_val = adx_aligned[i]
        vol_ratio = atr_ratio_aligned[i]
        
        if position == 0:
            # Long: RSI oversold (<30) + weak trend (ADX < 25) + low volatility regime (ATR ratio < 0.8)
            if (rsi_val < 30 and 
                adx_val < 25 and
                vol_ratio < 0.8):
                position = 1
                signals[i] = position_size
            # Short: RSI overbought (>70) + weak trend (ADX < 25) + low volatility regime (ATR ratio < 0.8)
            elif (rsi_val > 70 and 
                  adx_val < 25 and
                  vol_ratio < 0.8):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: RSI returns to neutral (40-60) or trend strengthens (ADX >= 25)
            if (rsi_val >= 40 and rsi_val <= 60) or adx_val >= 25:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: RSI returns to neutral (40-60) or trend strengthens (ADX >= 25)
            if (rsi_val >= 40 and rsi_val <= 60) or adx_val >= 25:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1h_4h_1d_RSI_ADX_ATR_Ratio_MeanReversion"
timeframe = "1h"
leverage = 1.0