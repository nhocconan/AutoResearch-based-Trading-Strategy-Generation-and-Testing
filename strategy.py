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
    
    # Get 4h and 1d data for trend and regime
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # 4h EMA34 for trend direction
    close_4h = df_4h['close'].values
    ema34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema34_4h)
    
    # 1d Donchian20 for volatility regime
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    donchian_high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donchian_width = donchian_high_20 - donchian_low_20
    donchian_width_ma = pd.Series(donchian_width).rolling(window=10, min_periods=10).mean().values
    donchian_width_ratio = donchian_width / donchian_width_ma
    donchian_width_ratio_aligned = align_htf_to_ltf(prices, df_1d, donchian_width_ratio)
    
    # 1h RSI14 for entry timing
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Time filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(100, n):
        # Skip if data not ready
        if np.isnan(ema34_4h_aligned[i]) or np.isnan(donchian_width_ratio_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 4h EMA34
        trend_up = close[i] > ema34_4h_aligned[i]
        trend_down = close[i] < ema34_4h_aligned[i]
        
        # Volatility regime: only trade in low volatility (width ratio < 0.8)
        low_vol = donchian_width_ratio_aligned[i] < 0.8
        
        # Time filter
        in_session = (8 <= hours[i] <= 20)
        
        if position == 0:
            # Long: uptrend + low vol + RSI oversold
            if trend_up and low_vol and in_session and rsi[i] < 30:
                signals[i] = 0.20
                position = 1
            # Short: downtrend + low vol + RSI overbought
            elif trend_down and low_vol and in_session and rsi[i] > 70:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Exit long: trend reversal or RSI overbought
            if not trend_up or rsi[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: trend reversal or RSI oversold
            if not trend_down or rsi[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_EMA34_Trend_DonchianVol_RSI_Entry"
timeframe = "1h"
leverage = 1.0