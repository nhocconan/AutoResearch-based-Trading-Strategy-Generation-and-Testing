#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d ADX Trend with Volume and Price Action Confirmation
# Uses ADX(14) to detect trending markets (>25) and ranges (<20) with hysteresis.
# In trending mode: buy pullbacks to EMA20 in uptrend, sell rallies to EMA20 in downtrend.
# In ranging mode: mean revert at Bollinger Bands (20,2) with RSI confirmation.
# Volume must be >1.3x average for all entries to ensure institutional participation.
# Designed for low trade frequency (<30/year) to minimize fee drag in both bull and bear markets.

name = "1d_ADXTrend_Range_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # === Price data ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === ADX calculation (14) ===
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    tr = np.maximum(high[1:] - low[1:], np.absolute(high[1:] - close[:-1]), np.absolute(low[1:] - close[:-1]))
    
    # Pad arrays to match length
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    tr = np.concatenate([[0], tr])
    
    # Smoothed values
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di = 100 * (pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / np.where(atr > 0, atr, np.nan))
    minus_di = 100 * (pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / np.where(atr > 0, atr, np.nan))
    dx = 100 * np.abs(plus_di - minus_di) / np.where((plus_di + minus_di) > 0, (plus_di + minus_di), np.nan)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # === EMA20 for trend following ===
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # === Bollinger Bands for mean reversion ===
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    bb_upper = sma_20 + (2 * std_20)
    bb_lower = sma_20 - (2 * std_20)
    
    # === RSI(14) for mean reversion confirmation ===
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss > 0, avg_gain / avg_loss, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.concatenate([[np.nan], rsi])  # align with price
    
    # === Volume confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, np.nan)
    
    # === Weekly trend filter (EMA20) ===
    weekly_close = df_1w['close'].values
    weekly_ema_20 = pd.Series(weekly_close).ewm(span=20, adjust=False, min_periods=20).mean().values
    weekly_ema_20_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Get values
        adx_val = adx[i]
        plus_di_val = plus_di[i]
        minus_di_val = minus_di[i]
        ema_val = ema_20[i]
        bb_upper_val = bb_upper[i]
        bb_lower_val = bb_lower[i]
        rsi_val = rsi[i]
        vol_ratio_val = vol_ratio[i]
        weekly_ema_val = weekly_ema_20_aligned[i]
        close_val = close[i]
        
        # Skip if any critical value is NaN
        if (np.isnan(adx_val) or np.isnan(ema_val) or np.isnan(bb_upper_val) or 
            np.isnan(bb_lower_val) or np.isnan(rsi_val) or np.isnan(vol_ratio_val) or
            np.isnan(weekly_ema_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Determine market regime using ADX with hysteresis
            # Trending: ADX > 25, Ranging: ADX < 20, Transition: hold previous state
            if i == 20:
                regime = 'trending' if adx_val > 25 else 'ranging'
            else:
                # Use hysteresis: stay in current regime unless ADX crosses opposite threshold
                if adx_val > 25:
                    regime = 'trending'
                elif adx_val < 20:
                    regime = 'ranging'
                else:
                    regime = regime  # hold previous regime
            
            if regime == 'trending':
                # Trend following: buy pullbacks in uptrend, sell rallies in downtrend
                if plus_di_val > minus_di_val and close_val > ema_val and vol_ratio_val > 1.3:
                    # Uptrend: wait for pullback to EMA
                    if close_val <= ema_val * 1.01:  # within 1% of EMA
                        signals[i] = 0.25
                        position = 1
                elif minus_di_val > plus_di_val and close_val < ema_val and vol_ratio_val > 1.3:
                    # Downtrend: wait for rally to EMA
                    if close_val >= ema_val * 0.99:  # within 1% of EMA
                        signals[i] = -0.25
                        position = -1
            else:  # ranging
                # Mean reversion at Bollinger Bands with RSI confirmation
                if close_val <= bb_lower_val and rsi_val < 30 and vol_ratio_val > 1.3:
                    signals[i] = 0.25
                    position = 1
                elif close_val >= bb_upper_val and rsi_val > 70 and vol_ratio_val > 1.3:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Long exit: trend breakdown or mean reversion signal
            if plus_di_val < minus_di_val or close_val >= bb_upper_val or rsi_val > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: trend breakdown or mean reversion signal
            if minus_di_val < plus_di_val or close_val <= bb_lower_val or rsi_val < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals