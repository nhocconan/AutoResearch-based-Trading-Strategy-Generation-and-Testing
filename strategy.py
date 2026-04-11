#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot breakout with 4h/1d trend filter and volume confirmation
# - Uses 4h trend (EMA50 > EMA200) and 1d trend (close > EMA50) for directional bias
# - 1h entries: long when price breaks above Camarilla H3 with volume > 1.3x 20-bar avg
# - Short when price breaks below Camarilla L3 with volume > 1.3x 20-bar avg
# - Session filter: 08-20 UTC to avoid low-volume Asian session noise
# - Exit: price returns to Camarilla pivot point (H3/L3 for stop, P for target)
# - Position size: 0.20 (20%) to control drawdown in volatile markets
# - Target: 15-25 trades/year (60-100 total over 4 years) to minimize fee drag
# - Works in bull markets by capturing breakouts with trend alignment
# - Works in bear markets by using 4h/1d trend filters to avoid counter-trend entries

name = "1h_4h_1d_camarilla_breakout_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Pre-compute session hours (08-20 UTC) - prices.index is DatetimeIndex
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 4h data ONCE before loop for EMA trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 200:
        return signals
    
    # Pre-compute 4h EMA50 and EMA200 for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200_4h = pd.Series(close_4h).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    ema_200_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_200_4h)
    
    # Load 1d data ONCE before loop for additional trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return signals
    
    # Pre-compute 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Pre-compute 1h Camarilla pivot levels (based on previous day's OHLC)
    # Camarilla levels: H4 = close + 1.5*(high-low), H3 = close + 1.1*(high-low), etc.
    # We use daily OHLC to calculate levels for intraday trading
    # For 1h chart, we need to align daily pivot levels to each 1h bar
    
    # Get daily OHLC
    df_1d_ohlc = get_htf_data(prices, '1d')
    if len(df_1d_ohlc) < 1:
        return signals
    
    # Calculate Camarilla levels from daily OHLC
    high_1d = df_1d_ohlc['high'].values
    low_1d = df_1d_ohlc['low'].values
    close_1d = df_1d_ohlc['close'].values
    
    # Camarilla levels
    H3 = close_1d + 1.1 * (high_1d - low_1d)
    L3 = close_1d - 1.1 * (high_1d - low_1d)
    H4 = close_1d + 1.5 * (high_1d - low_1d)
    L4 = close_1d - 1.5 * (high_1d - low_1d)
    P = (high_1d + low_1d + close_1d) / 3  # Pivot point
    
    # Align daily Camarilla levels to 1h timeframe
    H3_aligned = align_htf_to_ltf(prices, df_1d_ohlc, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d_ohlc, L3)
    H4_aligned = align_htf_to_ltf(prices, df_1d_ohlc, H4)
    L4_aligned = align_htf_to_ltf(prices, df_1d_ohlc, L4)
    P_aligned = align_htf_to_ltf(prices, df_1d_ohlc, P)
    
    # Pre-compute 1h volume confirmation (20-period average)
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(ema_200_4h_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(H3_aligned[i]) or
            np.isnan(L3_aligned[i]) or np.isnan(P_aligned[i]) or
            np.isnan(volume_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade during 08-20 UTC
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        volume_current = volume[i]
        
        # Camarilla levels
        h3 = H3_aligned[i]
        l3 = L3_aligned[i]
        h4 = H4_aligned[i]
        l4 = L4_aligned[i]
        p = P_aligned[i]
        
        # Volume confirmation: current volume > 1.3x 20-period average
        vol_confirm = volume_current > 1.3 * volume_sma_20[i]
        
        # 4h trend filter: EMA50 > EMA200 for uptrend, EMA50 < EMA200 for downtrend
        trend_4h_up = ema_50_4h_aligned[i] > ema_200_4h_aligned[i]
        trend_4h_down = ema_50_4h_aligned[i] < ema_200_4h_aligned[i]
        
        # 1d trend filter: close > EMA50 for long bias, close < EMA50 for short bias
        trend_1d_up = close_price > ema_50_1d_aligned[i]
        trend_1d_down = close_price < ema_50_1d_aligned[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long breakout: price above H3 with volume confirmation and trend alignment
        if close_price > h3 and vol_confirm and trend_4h_up and trend_1d_up:
            enter_long = True
        
        # Short breakout: price below L3 with volume confirmation and trend alignment
        if close_price < l3 and vol_confirm and trend_4h_down and trend_1d_down:
            enter_short = True
        
        # Exit conditions
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price returns to pivot point or hits H4 stop
            exit_long = close_price <= p or close_price >= h4
        elif position == -1:
            # Exit short if price returns to pivot point or hits L4 stop
            exit_short = close_price >= p or close_price <= l4
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.20
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.20
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.20 if position == 1 else (-0.20 if position == -1 else 0.0)
    
    return signals