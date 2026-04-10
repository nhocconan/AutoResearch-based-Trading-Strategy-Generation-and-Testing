#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h mean reversion with 4h/1d regime filter
# - Primary: 1h RSI(2) for short-term mean reversion
# - HTF: 4h ADX > 25 for trend regime + 1d close > 200 EMA for bull filter
# - Long: RSI(2) < 10 + 4h ADX > 25 + 1d close > 200 EMA
# - Short: RSI(2) > 90 + 4h ADX > 25 + 1d close < 200 EMA
# - Exit: RSI(2) crosses 50
# - Position sizing: 0.20 (discrete level to minimize fee churn)
# - Works in bull/bear: ADX filters trending markets, 200 EMA filters bull/bear bias, RSI(2) captures short-term reversals
# - Target: 80-120 trades over 4 years (20-30/year) to stay within fee drag limits

name = "1h_4h_1d_rsi2_adx_ema_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need enough data for calculations
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_4h) < 30 or len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 1h data
    close_1h = prices['close'].values
    
    # Pre-compute 4h data
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Pre-compute 1d data
    close_1d = df_1d['close'].values
    
    # Calculate 1h RSI(2)
    delta = pd.Series(close_1h).diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing for RSI
    alpha = 1.0 / 2
    avg_gain = pd.Series(gain).ewm(alpha=alpha, adjust=False, min_periods=2).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=alpha, adjust=False, min_periods=2).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi_2 = 100 - (100 / (1 + rs))
    
    # Calculate 4h ADX(14)
    # True Range
    tr1 = np.abs(np.roll(high_4h, 1) - np.roll(low_4h, 1))
    tr2 = np.abs(np.roll(high_4h, 1) - np.roll(close_4h, 1))
    tr3 = np.abs(np.roll(low_4h, 1) - np.roll(close_4h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = np.roll(high_4h, 1) - high_4h
    down_move = low_4h - np.roll(low_4h, 1)
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed values (Wilder's smoothing)
    period = 14
    alpha = 1.0 / period
    
    atr_4h = pd.Series(tr).ewm(alpha=alpha, adjust=False).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=alpha, adjust=False).mean().values / atr_4h
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=alpha, adjust=False).mean().values / atr_4h
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx_4h = pd.Series(dx).ewm(alpha=alpha, adjust=False).mean().values
    
    # Calculate 1d EMA(200)
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align all HTF indicators to 1h timeframe
    rsi_2_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close_1h}), rsi_2)  # 1h indicator, no alignment needed
    adx_4h_aligned = align_htf_to_ltf(prices, df_4h, adx_4h)
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(rsi_2[i]) or np.isnan(adx_4h_aligned[i]) or 
            np.isnan(ema_200_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            signals[i] = 0.0
            continue
        
        # Regime filters
        trend_filter = adx_4h_aligned[i] > 25.0
        bull_filter = close_1h[i] > ema_200_1d_aligned[i]
        bear_filter = close_1h[i] < ema_200_1d_aligned[i]
        
        # Mean reversion signals
        oversold = rsi_2[i] < 10.0
        overbought = rsi_2[i] > 90.0
        exit_signal = (rsi_2[i] > 50.0 and position == 1) or (rsi_2[i] < 50.0 and position == -1)
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Oversold + trend regime + bull bias
            if oversold and trend_filter and bull_filter:
                position = 1
                signals[i] = 0.20
            # Short entry: Overbought + trend regime + bear bias
            elif overbought and trend_filter and bear_filter:
                position = -1
                signals[i] = -0.20
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit: RSI crosses 50
            if exit_signal:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals