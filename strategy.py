#!/usr/bin/env python3
# 1h_Triple_Screen_RSI_MFI_Volume
# Hypothesis: 1h RSI(14) with MFI(14) confirmation and volume filter, gated by 4h EMA50 trend and 1d ADX trend strength.
# Uses 4h for trend direction, 1d for regime filter (ADX>25), and 1h for precise entry/exit.
# Designed for low trade frequency (15-30/year) to avoid fee drag in choppy 1h markets.
# Works in bull (trend-following) and bear (mean-reversion in range) via ADX regime filter.

name = "1h_Triple_Screen_RSI_MFI_Volume"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Get 1d data for ADX trend strength filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX(14)
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first bar
    
    plus_dm = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    minus_dm = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    plus_dm[0] = 0
    minus_dm[0] = 0
    
    tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    plus_dm14 = pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values
    minus_dm14 = pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values
    
    plus_di14 = 100 * plus_dm14 / tr14
    minus_di14 = 100 * minus_dm14 / tr14
    dx = 100 * np.abs(plus_di14 - minus_di14) / (plus_di14 + minus_di14)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # 1h indicators
    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # MFI(14)
    typical_price = (high + low + close) / 3.0
    raw_money_flow = typical_price * volume
    money_flow_pos = np.where(typical_price > np.roll(typical_price, 1), raw_money_flow, 0)
    money_flow_neg = np.where(typical_price < np.roll(typical_price, 1), raw_money_flow, 0)
    money_flow_pos[0] = 0
    money_flow_neg[0] = 0
    mf_ratio = pd.Series(money_flow_pos).rolling(window=14, min_periods=14).sum().values / \
               (pd.Series(money_flow_neg).rolling(window=14, min_periods=14).sum().values + 1e-10)
    mfi = 100 - (100 / (1 + mf_ratio))
    
    # Volume filter: current volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.3 * vol_ma)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(140, n):  # Start after warmup
        if not session_filter[i]:
            signals[i] = 0.0
            continue
            
        # Regime filter: ADX > 25 = trending, ADX < 20 = ranging
        if adx_aligned[i] > 25:
            # Trending regime: follow 4h EMA trend
            if position == 0:
                # LONG: RSI < 30 (oversold) + MFI < 30 + volume + price > 4h EMA
                if rsi[i] < 30 and mfi[i] < 30 and volume_filter[i] and close[i] > ema_4h_aligned[i]:
                    signals[i] = 0.20
                    position = 1
                # SHORT: RSI > 70 (overbought) + MFI > 70 + volume + price < 4h EMA
                elif rsi[i] > 70 and mfi[i] > 70 and volume_filter[i] and close[i] < ema_4h_aligned[i]:
                    signals[i] = -0.20
                    position = -1
                else:
                    signals[i] = 0.0
            elif position == 1:
                # EXIT LONG: RSI > 50 or price < 4h EMA
                if rsi[i] > 50 or close[i] < ema_4h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
            elif position == -1:
                # EXIT SHORT: RSI < 50 or price > 4h EMA
                if rsi[i] < 50 or close[i] > ema_4h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
        else:
            # Ranging regime: mean reversion at extremes
            if position == 0:
                # LONG: RSI < 20 + MFI < 20 + volume (deep oversold)
                if rsi[i] < 20 and mfi[i] < 20 and volume_filter[i]:
                    signals[i] = 0.20
                    position = 1
                # SHORT: RSI > 80 + MFI > 80 + volume (deep overbought)
                elif rsi[i] > 80 and mfi[i] > 80 and volume_filter[i]:
                    signals[i] = -0.20
                    position = -1
                else:
                    signals[i] = 0.0
            elif position == 1:
                # EXIT LONG: RSI > 60 (overbought in range)
                if rsi[i] > 60:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
            elif position == -1:
                # EXIT SHORT: RSI < 40 (oversold in range)
                if rsi[i] < 40:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
    
    return signals