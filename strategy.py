#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h momentum with 4h trend filter and 1d regime filter.
# Uses 1h RSI(14) for momentum signals, 4h EMA(50) for trend direction,
# and 1d ADX(14) for regime filtering (only trade when ADX > 25).
# In trending markets (ADX > 25): go long when RSI crosses above 50 and price > 4h EMA50,
# go short when RSI crosses below 50 and price < 4h EMA50.
# Volume confirmation: require volume > 1.3x 20-period average.
# Position size 0.20 for risk control. Session filter: trade only 08-20 UTC.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4h data (trend filter) ===
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # === 1d data (regime filter) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # === 4h EMA50 for trend direction ===
    close_4h_series = pd.Series(close_4h)
    ema50_4h = close_4h_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # === 1d ADX(14) for regime filter ===
    # Calculate True Range
    tr1 = pd.Series(high_1d).shift(1) - pd.Series(low_1d).shift(1)
    tr2 = abs(pd.Series(high_1d).shift(1) - pd.Series(close_1d).shift(1))
    tr3 = abs(pd.Series(low_1d).shift(1) - pd.Series(close_1d).shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = tr.rolling(window=14, min_periods=14).mean().values
    
    # Calculate Directional Movement
    up_move = pd.Series(high_1d).diff()
    down_move = -pd.Series(low_1d).diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smooth DM and TR
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / atr_1d
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / atr_1d
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # === 1h RSI(14) for momentum ===
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values  # Fill NaN with 50 (neutral)
    
    # === 1h volume ratio for confirmation ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma_20
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 100
    
    # Track position and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(rsi[i]) or np.isnan(ema50_4h_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        rsi_val = rsi[i]
        ema50_val = ema50_4h_aligned[i]
        adx_val = adx_aligned[i]
        vol_ratio_val = vol_ratio[i]
        hour = hours[i]
        
        # Session filter: only trade 08-20 UTC
        if not (8 <= hour <= 20):
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === STOPLOSS LOGIC ===
        if position == 1:  # Long position
            atr_1h = np.abs(high - low)
            atr_ma = pd.Series(atr_1h).rolling(window=14, min_periods=14).mean().values
            if price < entry_price - 2.0 * atr_ma[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        elif position == -1:  # Short position
            atr_1h = np.abs(high - low)
            atr_ma = pd.Series(atr_1h).rolling(window=14, min_periods=14).mean().values
            if price > entry_price + 2.0 * atr_ma[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when RSI falls below 40 or trend changes
            if rsi_val < 40 or price < ema50_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        elif position == -1:  # Short position
            # Exit when RSI rises above 60 or trend changes
            if rsi_val > 60 or price > ema50_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Require trending market (ADX > 25) and volume confirmation
            if adx_val > 25 and vol_ratio_val > 1.3:
                # Go long when RSI crosses above 50 and price above 4h EMA50
                if rsi_val > 50 and rsi[i-1] <= 50 and price > ema50_val:
                    signals[i] = 0.20
                    position = 1
                    entry_price = price
                    continue
                # Go short when RSI crosses below 50 and price below 4h EMA50
                elif rsi_val < 50 and rsi[i-1] >= 50 and price < ema50_val:
                    signals[i] = -0.20
                    position = -1
                    entry_price = price
                    continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.20
        elif position == -1:
            signals[i] = -0.20
        else:
            signals[i] = 0.0
    
    return signals

name = "1h_RSI_EMA50_ADX_Volume_Session"
timeframe = "1h"
leverage = 1.0