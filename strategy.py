#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4h data for direction and structure ===
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # 4h ATR for volatility filter and stop
    high_4h_series = pd.Series(high_4h)
    low_4h_series = pd.Series(low_4h)
    close_4h_series = pd.Series(close_4h)
    tr1 = high_4h_series - low_4h_series
    tr2 = abs(high_4h_series - close_4h_series.shift(1))
    tr3 = abs(low_4h_series - close_4h_series.shift(1))
    tr1.iloc[0] = np.inf
    tr2.iloc[0] = np.inf
    tr3.iloc[0] = np.inf
    tr_4h = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_4h = tr_4h.rolling(window=14, min_periods=14).mean().values
    atr_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_4h)
    
    # 4h EMA34 for trend filter
    ema_34_4h = close_4h_series.ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # === 1d data for regime filter ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 1d ADX for trend strength
    high_1d_series = pd.Series(high_1d)
    low_1d_series = pd.Series(low_1d)
    close_1d_series = pd.Series(close_1d)
    up_move = high_1d_series - high_1d_series.shift(1)
    down_move = low_1d_series.shift(1) - low_1d_series
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    tr1_1d = abs(high_1d_series - low_1d_series)
    tr2_1d = abs(high_1d_series - close_1d_series.shift(1))
    tr3_1d = abs(low_1d_series - close_1d_series.shift(1))
    tr1_1d.iloc[0] = np.inf
    tr2_1d.iloc[0] = np.inf
    tr3_1d.iloc[0] = np.inf
    tr_1d = pd.concat([tr1_1d, tr2_1d, tr3_1d], axis=1).max(axis=1)
    atr_1d = tr_1d.rolling(window=14, min_periods=14).mean()
    plus_di = 100 * (pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean() / atr_1d)
    minus_di = 100 * (pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean() / atr_1d)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)) * 100
    adx = dx.ewm(alpha=1/14, adjust=False).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # === 4h indicators for entry timing ===
    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume spike detection (2x average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma_20
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators have valid data
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_4h_aligned[i]) or np.isnan(atr_4h_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(rsi[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        price = close[i]
        ema_34_4h_val = ema_34_4h_aligned[i]
        atr_4h_val = atr_4h_aligned[i]
        adx_val = adx_aligned[i]
        rsi_val = rsi[i]
        vol_ratio_val = vol_ratio[i]
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when price closes below EMA34 OR RSI becomes overbought OR ATR-based stop
            if (price < ema_34_4h_val) or (rsi_val > 70) or (price < entry_price - 1.5 * atr_4h_val):
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit when price closes above EMA34 OR RSI becomes oversold OR ATR-based stop
            if (price > ema_34_4h_val) or (rsi_val < 30) or (price > entry_price + 1.5 * atr_4h_val):
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Only trade during session and when trend is strong enough (ADX > 25)
            if in_session and adx_val > 25:
                # LONG: Price above EMA34 AND RSI not overbought AND volume spike
                if (price > ema_34_4h_val) and (rsi_val < 60) and (vol_ratio_val > 2.0):
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
                    continue
                
                # SHORT: Price below EMA34 AND RSI not oversold AND volume spike
                elif (price < ema_34_4h_val) and (rsi_val > 40) and (vol_ratio_val > 2.0):
                    signals[i] = -0.25
                    position = -1
                    entry_price = price
                    continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_EMA34_ADX_RSI_Volume"
timeframe = "4h"
leverage = 1.0